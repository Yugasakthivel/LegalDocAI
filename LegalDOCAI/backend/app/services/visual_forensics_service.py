import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Any, List, Tuple

class AuthenticityCNN(nn.Module):
    """
    CNN Architecture for Document Authenticity Detection.
    Detects visual anomalies like copy-paste artifacts, font inconsistencies, and noise patterns.
    """
    def __init__(self):
        super(AuthenticityCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 2)  # Binary: 0=Fake, 1=Real
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 32 * 32)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class VisualForensicsService:
    def __init__(self):
        self.model = AuthenticityCNN()
        # In production, we would load pre-trained weights here.
        # self.model.load_state_dict(torch.load("data/models/authenticity_cnn.pth"))
        self.model.eval()

    def get_lbp_features(self, image: np.ndarray) -> float:
        """
        Generates a texture score using Local Binary Patterns (LBP).
        Detects unnatural smoothness or graininess associated with forgeries.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Basic LBP implementation using OpenCV
        # We look for texture consistency across the document
        h, w = gray.shape
        lbp_img = np.zeros((h-2, w-2), dtype=np.uint8)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp_img[i-1, j-1] = code
        
        # Calculate entropy of LBP histogram
        hist = cv2.calcHist([lbp_img], [0], None, [256], [0, 256])
        hist /= hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Normalized texture score (0.0 to 1.0)
        # Authentic documents usually have an entropy between 5.0 and 7.5
        score = max(0.0, min(1.0, entropy / 8.0))
        return float(score)

    def analyze_layout_consistency(self, image: np.ndarray) -> float:
        """
        Analyzes margin consistency and structural alignment.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Threshold to find text blocks
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find horizontal and vertical projections
        h_proj = np.sum(thresh, axis=1)
        v_proj = np.sum(thresh, axis=0)
        
        # Analyze margins (first and last non-zero pixels)
        h_indices = np.where(h_proj > 0)[0]
        v_indices = np.where(v_proj > 0)[0]
        
        if len(h_indices) == 0 or len(v_indices) == 0:
            return 0.5
            
        top_margin = h_indices[0]
        bottom_margin = gray.shape[0] - h_indices[-1]
        left_margin = v_indices[0]
        right_margin = gray.shape[1] - v_indices[-1]
        
        # Score based on "standard" legal margins (usually balanced)
        margin_diff = abs(left_margin - right_margin) / gray.shape[1]
        score = 1.0 - min(1.0, margin_diff * 5) # Penalize highly unbalanced margins
        
        return float(score)

    def detect_visual_fraud(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Unified visual forensics analysis.
        """
        try:
            # Prepare image
            img_cv = np.array(pil_image.convert("RGB"))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # 1. Texture Score (LBP)
            texture_score = self.get_lbp_features(img_cv)
            
            # 2. Layout Score
            layout_score = self.analyze_layout_consistency(img_cv)
            
            # 3. CNN Score (Placeholder logic for demo)
            # In production, we'd run:
            # img_tensor = transforms.ToTensor()(pil_image.convert("L").resize((256, 256))).unsqueeze(0)
            # output = self.model(img_tensor)
            # cnn_score = F.softmax(output, dim=1)[0][1].item()
            cnn_score = (texture_score + layout_score) / 2 # Simulated
            
            # Final Fusion Score
            final_visual_score = (texture_score * 0.3) + (layout_score * 0.3) + (cnn_score * 0.4)
            
            return {
                "visual_score": round(final_visual_score * 100, 2),
                "sub_scores": {
                    "texture_consistency": round(texture_score * 100, 2),
                    "layout_consistency": round(layout_score * 100, 2),
                    "cnn_authenticity": round(cnn_score * 100, 2)
                },
                "verdict": "Real" if final_visual_score > 0.7 else "Suspicious" if final_visual_score > 0.4 else "Fake"
            }
        except Exception as e:
            return {"error": f"Visual forensics failed: {str(e)}", "visual_score": 50}

visual_forensics = VisualForensicsService()
