import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np
import nibabel as nib
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import json
from rapidfuzz import fuzz, process
from pathlib import Path
from torchvision.models import densenet121, DenseNet121_Weights, resnet50, ResNet50_Weights
import timm
from sklearn.preprocessing import MinMaxScaler
import tempfile

# Streamlit page config
st.set_page_config(page_title="Alzheimer’s Assistant", layout="wide")

# Device
device = torch.device("cpu")
st.write(f"Using device: {device}")

# Model Definitions
class MultimodalMRIModel(nn.Module):
    def __init__(self, num_classes=3, clinical_dim=3):
        super(MultimodalMRIModel, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.resnet.fc = nn.Identity()
        
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, clinical):
        img_features = self.resnet(images)
        clin_features = self.clinical_mlp(clinical)
        combined = torch.cat((img_features, clin_features), dim=1)
        output = self.fc(combined)
        return output

class MultimodalDenseNet(nn.Module):
    def __init__(self, num_classes=3, clinical_dim=3):
        super(MultimodalDenseNet, self).__init__()
        self.densenet = densenet121(weights=DenseNet121_Weights.DEFAULT)
        self.densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.densenet.features.conv0.weight, mode='fan_out', nonlinearity='relu')
        self.densenet.classifier = nn.Identity()
        
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(1024 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, images, clinical):
        img_features = self.densenet(images)
        clin_features = self.clinical_mlp(clinical)
        combined = torch.cat((img_features, clin_features), dim=1)
        output = self.fc(combined)
        return output

class MultimodalEfficientNet(nn.Module):
    def __init__(self, num_classes=3, clinical_dim=3):
        super(MultimodalEfficientNet, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=True, in_chans=1)
        for param in self.efficientnet.conv_stem.parameters():
            param.requires_grad = False
        for param in self.efficientnet.bn1.parameters():
            param.requires_grad = False
        self.efficientnet.classifier = nn.Identity()
        self.feature_dim = 1280

        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1280 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, clinical):
        img_features = self.efficientnet(images)
        clin_features = self.clinical_mlp(clinical)
        combined = torch.cat((img_features, clin_features), dim=1)
        output = self.fc(combined)
        return output

# Load models
@st.cache_resource
def load_models():
    resnet_model = MultimodalMRIModel(num_classes=3, clinical_dim=3)
    densenet_model = MultimodalDenseNet(num_classes=3, clinical_dim=3)
    efficientnet_model = MultimodalEfficientNet(num_classes=3, clinical_dim=3)
    models = [resnet_model, densenet_model, efficientnet_model]
    weights_paths = [
        "best_multimodal_mri.pth",
        "best_multimodal_densenet.pth",
        "best_multimodal_efficientnet.pth"
    ]
    try:
        for model, weight_path in zip(models, weights_paths):
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
            model.to(device)
            model.eval()
        st.success("Loaded all model weights")
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        raise

models = load_models()

# Chatbot Implementation
def train_chatbot():
    try:
        with open("knowledge_base.json", "r") as f:
            knowledge_base = json.load(f)
    except FileNotFoundError:
        st.warning("No knowledge_base.json found! Creating new one...")
        knowledge_base = [
            {
                "question": "What is Alzheimer’s disease?",
                "answer": "Alzheimer’s is a progressive brain disorder that impairs memory, thinking, and behavior, often starting with subtle forgetfulness and advancing to severe dependency. It’s the leading cause of dementia, affecting millions, with no cure yet. Resources like 'The 36-Hour Day' explain its impact on families.",
                "keywords": ["alzheimer’s", "what is", "disease", "dementia", "memory", "brain"],
                "relevance": 1.0
            },
            {
                "question": "What are the symptoms of Alzheimer’s?",
                "answer": "Early symptoms include forgetting recent events, difficulty with names, and misplacing items, while later stages bring confusion, mood swings, and trouble speaking. 'The 36-Hour Day' notes that symptoms vary, but routines can help manage them.",
                "keywords": ["symptoms", "alzheimer’s", "signs", "memory loss", "confusion", "behavior"],
                "relevance": 1.0
            },
            {
                "question": "How is Alzheimer’s diagnosed?",
                "answer": "Diagnosis involves assessing medical history, cognitive tests like memory quizzes, and brain scans (MRI or PET) to detect changes. Blood tests may rule out other causes. 'Alzheimer’s Disease: What If There Was a Cure?' emphasizes early diagnosis for better planning.",
                "keywords": ["diagnosed", "alzheimer’s", "tests", "cognitive", "brain scans", "diagnosis", "how"],
                "relevance": 0.95
            },
            {
                "question": "What causes Alzheimer’s disease?",
                "answer": "Alzheimer’s stems from brain cell damage due to amyloid plaques and tau tangles, influenced by genetics, aging, and lifestyle factors like diet. 'Alzheimer’s Disease: What If There Was a Cure?' explores ketones as a potential aid.",
                "keywords": ["causes", "alzheimer’s", "why", "plaques", "tangles", "genetics"],
                "relevance": 0.9
            },
            {
                "question": "What treatments are available for Alzheimer’s?",
                "answer": "Medications like donepezil or memantine can ease symptoms, but there’s no cure. Lifestyle changes, such as exercise and social engagement, support brain health. 'The 36-Hour Day' suggests combining medical and emotional care.",
                "keywords": ["treatments", "alzheimer’s", "medications", "manage", "therapy", "drugs"],
                "relevance": 0.95
            },
            {
                "question": "Is there a cure for Alzheimer’s?",
                "answer": "No cure exists, but research into drugs and diets, like ketones discussed in 'Alzheimer’s Disease: What If There Was a Cure?', offers hope. Always consult a doctor for updates.",
                "keywords": ["cure", "alzheimer’s", "healed", "treatment", "remedy"],
                "relevance": 0.85
            },
            {
                "question": "How can I support someone with Alzheimer’s?",
                "answer": "Offer patience, simplify tasks, and maintain a calm environment. 'The 36-Hour Day' advises learning their needs and joining support groups for caregivers.",
                "keywords": ["support", "alzheimer’s", "care", "caregiving", "help", "patient"],
                "relevance": 1.0
            },
            {
                "question": "How to communicate with someone who has Alzheimer’s?",
                "answer": "Use short sentences, speak slowly, and focus on nonverbal cues like smiles. 'Learning to Speak Alzheimer’s' teaches ‘habilitation,’ prioritizing emotional connection over correction.",
                "keywords": ["communicate", "alzheimer’s", "talk", "speak", "patient", "conversation"],
                "relevance": 0.95
            },
            {
                "question": "What are the early signs of Alzheimer’s?",
                "answer": "Early signs include trouble remembering conversations, losing track of dates, or struggling with decisions. Unlike normal aging, these disrupt daily life, per 'The 36-Hour Day'.",
                "keywords": ["early signs", "alzheimer’s", "first symptoms", "forgetting", "memory"],
                "relevance": 0.95
            },
            {
                "question": "Can Alzheimer’s be prevented?",
                "answer": "Prevention isn’t guaranteed, but a Mediterranean diet, exercise, and mental activities may reduce risk. 'Alzheimer’s Disease: What If There Was a Cure?' highlights brain-healthy lifestyles.",
                "keywords": ["prevent", "alzheimer’s", "avoid", "reduce risk", "healthy", "lifestyle"],
                "relevance": 0.9
            },
            {
                "question": "Is Alzheimer’s hereditary?",
                "answer": "Genetics, like the APOE4 gene, can raise risk, especially in early-onset cases, but most Alzheimer’s isn’t purely inherited. Lifestyle matters too, per 'The 36-Hour Day'.",
                "keywords": ["hereditary", "alzheimer’s", "genetic", "family", "inherited", "genes"],
                "relevance": 0.9
            },
            {
                "question": "What’s the difference between Alzheimer’s and dementia?",
                "answer": "Dementia describes symptoms like memory loss affecting daily life; Alzheimer’s is a disease causing most dementia cases. 'Learning to Speak Alzheimer’s' clarifies this for families.",
                "keywords": ["difference", "alzheimer’s", "dementia", "cognitive", "memory"],
                "relevance": 0.9
            },
            {
                "question": "What are the stages of Alzheimer’s?",
                "answer": "Alzheimer’s progresses from mild (forgetfulness), to moderate (difficulty with tasks), to severe (needing full-time care). 'The 36-Hour Day' details how families adapt at each stage.",
                "keywords": ["stages", "alzheimer’s", "progression", "mild", "severe", "moderate"],
                "relevance": 0.85
            },
            {
                "question": "How does Alzheimer’s affect daily life?",
                "answer": "It impairs memory and judgment, making tasks like shopping or cooking hard. 'The 36-Hour Day' suggests structured routines to ease challenges for patients.",
                "keywords": ["daily life", "alzheimer’s", "affect", "tasks", "routine", "independence"],
                "relevance": 0.9
            },
            {
                "question": "Where can I find help for Alzheimer’s?",
                "answer": "Alz.org offers a 24/7 helpline (1-800-272-3900), support groups, and care tips. 'The 36-Hour Day' recommends local Alzheimer’s associations for personalized help.",
                "keywords": ["help", "alzheimer’s", "resources", "support groups", "helpline"],
                "relevance": 1.0
            },
            {
                "question": "What is early-onset Alzheimer’s?",
                "answer": "Early-onset Alzheimer’s strikes before age 65, often tied to genetic mutations, with faster progression. 'The 36-Hour Day' notes its unique challenges for younger families.",
                "keywords": ["early-onset", "alzheimer’s", "young", "genetic", "under 65"],
                "relevance": 0.85
            },
            {
                "question": "How to manage caregiver stress for Alzheimer’s?",
                "answer": "Take breaks, seek respite care, and join support groups to share experiences. 'The 36-Hour Day' emphasizes self-care to sustain caregiving energy.",
                "keywords": ["caregiver", "alzheimer’s", "stress", "support", "caregiving"],
                "relevance": 0.95
            },
            {
                "question": "What are Alzheimer’s risk factors?",
                "answer": "Age, family history, and genes like APOE4 increase risk, as do poor diet and inactivity. 'Alzheimer’s Disease: What If There Was a Cure?' stresses modifiable factors like exercise.",
                "keywords": ["risk factors", "alzheimer’s", "causes", "age", "genetics", "lifestyle"],
                "relevance": 0.9
            },
            {
                "question": "How to plan for Alzheimer’s care?",
                "answer": "Plan finances, legal documents, and home safety early. 'The 36-Hour Day' advises involving family and professionals to tailor care as needs grow.",
                "keywords": ["plan", "alzheimer’s", "care", "planning", "safety", "family"],
                "relevance": 0.9
            },
            {
                "question": "What is the role of diet in Alzheimer’s?",
                "answer": "Diets rich in fruits, vegetables, and omega-3s may support brain health, while ketones are being studied for benefits. 'Alzheimer’s Disease: What If There Was a Cure?' explores dietary impacts.",
                "keywords": ["diet", "alzheimer’s", "food", "nutrition", "ketones", "brain"],
                "relevance": 0.85
            },
            {
                "question": "How to handle Alzheimer’s agitation?",
                "answer": "Stay calm, avoid triggers, and use soothing activities like music. 'Learning to Speak Alzheimer’s' suggests redirecting attention gently to reduce distress.",
                "keywords": ["agitation", "alzheimer’s", "behavior", "calm", "manage"],
                "relevance": 0.9
            },
            {
                "question": "What are myths about Alzheimer’s?",
                "answer": "Myths include ‘it’s just aging’ or ‘only old people get it.’ 'The 36-Hour Day' debunks these, noting Alzheimer’s is a disease with diverse impacts.",
                "keywords": ["myths", "alzheimer’s", "misconceptions", "aging", "disease"],
                "relevance": 0.85
            }
        ]
        with open("knowledge_base.json", "w") as f:
            json.dump(knowledge_base, f, indent=4)
    return knowledge_base

def alzheimer_chatbot_response(user_input, knowledge_base):
    if user_input.lower().strip() == "quit":
        return "Goodbye! Stay informed and take care."
    
    best_match = None
    max_score = 0
    
    for entry in knowledge_base:
        keyword_score = 0
        input_words = user_input.lower().split()
        for keyword in entry["keywords"]:
            best_match_word = process.extractOne(keyword, input_words, scorer=fuzz.token_sort_ratio)
            if best_match_word and best_match_word[1] >= 80:
                keyword_score += best_match_word[1] / 100
        if "alzheimer" in user_input.lower() or "dementia" in user_input.lower():
            score = keyword_score * entry["relevance"] * 1.3
        elif any(fuzz.partial_ratio(term, user_input.lower()) >= 80 for term in ["diagnosed", "diagnosis", "symptoms", "signs", "care", "caregiving"]):
            score = keyword_score * entry["relevance"] * 1.1
        else:
            score = keyword_score * entry["relevance"] * 0.8
        if score > max_score:
            max_score = score
            best_match = entry
    
    if best_match and max_score >= 0.5:
        return best_match["answer"]
    else:
        return "I’m not sure I understood. Try asking about Alzheimer’s symptoms, care, or myths. For example, try 'How is Alzheimer’s diagnosed?' or 'How to support someone with Alzheimer’s?'"

# Initialize chatbot
knowledge_base = train_chatbot()

# Helper Functions
def process_nii_to_png(nii_path, output_path, slice_idx=90):
    try:
        img = nib.load(nii_path).get_fdata()
        if slice_idx >= img.shape[2]:
            slice_idx = img.shape[2] // 2
        slice_data = img[:, :, slice_idx]
        slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min() + 1e-8)
        slice_data = cv2.resize(slice_data, (224, 224))
        slice_data = (slice_data * 255).astype(np.uint8)
        cv2.imwrite(output_path, slice_data)
        return True
    except Exception as e:
        st.error(f"Error processing .nii file: {e}")
        return False

def process_clinical_data(age, sex):
    try:
        # Normalize age (same scaler as training)
        scaler = MinMaxScaler()
        # Fit scaler on a dummy range (since we don’t have the original data)
        # Assuming age range 50-90 based on Alzheimer’s data
        scaler.fit([[50], [90]])
        age_normalized = scaler.transform([[float(age)]])[0][0]
        
        # Encode sex
        sex_m = 1.0 if sex == "Male" else 0.0
        sex_f = 1.0 if sex == "Female" else 0.0
        
        clinical = np.array([age_normalized, sex_m, sex_f], dtype=np.float32)
        return clinical
    except Exception as e:
        st.error(f"Error processing clinical data: {e}")
        return None

def predict_single_image(img_path, clinical_data, models, transform, device):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    if np.max(img) == 0:
        raise ValueError("Blank image")
    
    augmented = transform(image=img)
    img = augmented['image']
    img = img.unsqueeze(0).to(device)
    
    if clinical_data.shape != (3,) or np.any(np.isnan(clinical_data)):
        raise ValueError("Clinical data must be 3 values without NaNs")
    clinical = torch.tensor(clinical_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = []
        for model in models:
            pred = model(img, clinical)
            probs.append(torch.softmax(pred, dim=1))
        avg_probs = sum(probs) / len(probs)
        pred_class = torch.argmax(avg_probs, dim=1).item()
        confidence = avg_probs[0, pred_class].item()
    
    class_names = {0: "Alzheimer’s Disease (AD)", 1: "Mild Cognitive Impairment (MCI)", 2: "Cognitively Normal (CN)"}
    return class_names[pred_class], confidence

# Test transform
test_transform = A.Compose([
    A.Normalize(mean=[0.0], std=[1.0]),
    ToTensorV2()
])

# Streamlit UI
st.title("Alzheimer’s Assistant")
st.markdown("""
**Disclaimer:** This tool provides general information and predictions for research purposes. It is not a substitute for professional medical advice. Consult a doctor for diagnosis or treatment.
""")

# Layout with two columns
col1, col2 = st.columns(2)

# MRI Classification Section
with col1:
    st.header("MRI Classification")
    with st.form(key="predict_form"):
        nii_file = st.file_uploader("Upload MRI Image (.nii or .nii.gz)", type=["nii", "nii.gz"])
        age = st.number_input("Age", min_value=50, max_value=90, value=70)
        sex = st.selectbox("Sex", ["Male", "Female"])
        submit_button = st.form_submit_button("Predict")
    
    if submit_button and nii_file:
        with st.spinner("Processing..."):
            # Save uploaded .nii file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as tmp:
                tmp.write(nii_file.read())
                nii_path = tmp.name
            
            # Convert .nii to PNG
            output_dir = "mri_png"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{nii_file.name.split('.')[0]}.png")
            if process_nii_to_png(nii_path, output_path):
                # Process clinical data
                clinical_data = process_clinical_data(age, sex)
                if clinical_data is not None:
                    try:
                        prediction, confidence = predict_single_image(output_path, clinical_data, models, test_transform, device)
                        st.success(f"**Prediction:** {prediction} (Confidence: {confidence:.4f})")
                        # Display PNG
                        st.image(output_path, caption="Processed MRI Slice", use_column_width=True)
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
                    finally:
                        # Clean up
                        os.remove(nii_path)
                        os.remove(output_path) if os.path.exists(output_path) else None
                else:
                    os.remove(nii_path)
            else:
                os.remove(nii_path)

# Chatbot Section
with col2:
    st.header("Alzheimer’s Chatbot")
    st.write("Ask questions about Alzheimer’s disease. Type 'quit' to end the session.")
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Chat input
    user_input = st.text_input("Your question:", key="chat_input")
    if st.button("Send") and user_input:
        response = alzheimer_chatbot_response(user_input, knowledge_base)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for sender, message in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"<div style='text-align: right; background-color: #007bff; color: white; padding: 10px; border-radius: 10px; margin: 5px;'>{sender}: {message}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='text-align: left; background-color: #e9ecef; color: black; padding: 10px; border-radius: 10px; margin: 5px;'>{sender}: {message}</div>", unsafe_allow_html=True)

# Custom CSS for styling
st.markdown("""
<style>
.stTextInput > div > div > input {
    width: 100%;
}
.stButton > button {
    background-color: #007bff;
    color: white;
}
.stButton > button:hover {
    background-color: #0056b3;
}
</style>
""", unsafe_allow_html=True)