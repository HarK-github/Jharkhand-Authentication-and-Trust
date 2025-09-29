"""
Multimodal Forgery Detection - Layer 2
=====================================

Components implemented:
 - Seal / Stamp verification: Siamese network (PyTorch) for one-shot/verification.
 - Font & Text Inconsistency Analysis: ELA-based detector + text-region feature extractor using Tesseract OCR to localize text boxes; a scikit-learn classifier to detect anomalies.
 - Signature Verification: Siamese network (PyTorch) similar to seal verification.

This file provides training scaffolding, dataset classes, inference helpers, and a simple demo usage. It is intended as a starting point and needs
real datasets (genuine vs forged images) and hyperparameter tuning for production.

Dependencies:
 - python >= 3.8
 - pytorch
 - torchvision
 - opencv-python
 - pillow
 - pytesseract (Tesseract OCR must be installed separately on the system)
 - scikit-learn
 - numpy
 - scikit-image

Notes:
 - Replace dataset paths with your datasets containing labeled images (genuine/forged). For siamese training you need pairs: (img1, img2, label)
 - ELA (Error Level Analysis) requires saving a compressed JPEG and comparing pixel-level differences; we implement a PIL-based approach.
 - This code focuses on clarity and modularity rather than maximum training speed or production robustness.

"""

import os
import io
import math
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageChops
import cv2
import pytesseract

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision import models

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ----------------------------- Utilities ---------------------------------

def ensure_rgb(img: Image.Image) -> Image.Image:
    if img.mode == 'RGB':
        return img
    return img.convert('RGB')


def load_image(path: str) -> Image.Image:
    return ensure_rgb(Image.open(path))


# ----------------------------- ELA Functions -----------------------------

def error_level_analysis(pil_img: Image.Image, quality: int = 90) -> Image.Image:
    """Compute ELA image: save with a given JPEG quality and subtract from original.
    Returns a PIL Image (RGB) where higher pixel differences indicate tampering.
    """
    pil_img = ensure_rgb(pil_img)
    with io.BytesIO() as buffer:
        pil_img.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)
        compressed = ensure_rgb(compressed)

    diff = ImageChops.difference(pil_img, compressed)

    # scale difference to make it more visible
    extrema = diff.getextrema()
    max_diff = max([e[1] for e in extrema]) if extrema else 1
    if max_diff == 0:
        max_diff = 1
    scale = int(255.0 / max_diff)
    ela_img = ImageEnhance = None
    try:
        from PIL import ImageEnhance
        enhancer = ImageEnhance.Brightness(diff)
        ela_img = enhancer.enhance(scale)
    except Exception:
        # fallback: multiply channels
        ela_img = diff.point(lambda p: p * scale)

    return ela_img


def ela_features_from_box(pil_img: Image.Image, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Extract simple ELA + texture features from a bounding box (left, top, width, height)
    Features returned:
      - mean, std, max of ELA image channels
      - edge density (Canny) within box
      - compression artifact signature: variance of Laplacian
    """
    left, top, w, h = box
    crop = pil_img.crop((left, top, left + w, top + h)).resize((224, 224))
    ela_img = error_level_analysis(crop, quality=90).convert('L')
    arr = np.array(ela_img).astype(np.float32) / 255.0

    mean = arr.mean()
    std = arr.std()
    mx = arr.max()
    mn = arr.min()

    # edge density
    edges = cv2.Canny((arr * 255).astype(np.uint8), 50, 150)
    edge_density = edges.sum() / (edges.size + 1e-9)

    # Laplacian variance (blurriness / resampling artifact indicator)
    lap = cv2.Laplacian((arr * 255).astype(np.uint8), cv2.CV_64F)
    lap_var = float(lap.var())

    return np.array([mean, std, mx, mn, edge_density, lap_var], dtype=np.float32)


# ----------------------------- OCR Helpers -------------------------------

def ocr_text_boxes(pil_img: Image.Image) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """Return list of (text, (left, top, width, height)) using pytesseract's boxes.
    Note: pytesseract.image_to_data produces bounding boxes per word.
    """
    img = np.array(pil_img)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    boxes = []
    n_boxes = len(data['level'])
    for i in range(n_boxes):
        text = data['text'][i].strip()
        if text == '' or text.isspace():
            continue
        left = int(data['left'][i])
        top = int(data['top'][i])
        w = int(data['width'][i])
        h = int(data['height'][i])
        boxes.append((text, (left, top, w, h)))
    return boxes


# ----------------------------- Siamese Network ---------------------------

class EmbeddingNet(nn.Module):
    """Simple embedding network based on a small ResNet backbone.
    We'll use a pre-trained ResNet18 and adapt it for embeddings.
    """
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(pretrained=pretrained)
        layers = list(resnet.children())[:-1]  # drop fc
        self.backbone = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.backbone(x)  # Bx512x1x1
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = nn.functional.normalize(x, dim=1)
        return x


class SiameseNet(nn.Module):
    def __init__(self, embedding_net: EmbeddingNet):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        e1 = self.embedding_net(x1)
        e2 = self.embedding_net(x2)
        return e1, e2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, e1, e2, label):
        # label: 1 if similar (genuine pair), 0 if dissimilar
        dist = (e1 - e2).pow(2).sum(1)
        loss_sim = label * dist
        loss_dis = (1 - label) * torch.clamp(self.margin - torch.sqrt(dist + 1e-9), min=0.0).pow(2)
        loss = 0.5 * (loss_sim + loss_dis).mean()
        return loss


# ----------------------------- Dataset classes --------------------------

class SiamesePairDataset(Dataset):
    """Dataset that yields pairs (img1, img2, label).
    Expects directories structured like:
      root/genuine/...
      root/forged/...
    You'll need to provide a strategy for sampling positive and negative pairs.
    """
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.classes = []
        # gather files
        self.genuine = []
        self.forged = []
        for cls in ['genuine', 'forged']:
            p = os.path.join(root, cls)
            if not os.path.isdir(p):
                continue
            files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if cls == 'genuine':
                self.genuine.extend(files)
            else:
                self.forged.extend(files)

        self.all_files = self.genuine + self.forged

    def __len__(self):
        return max(1000, len(self.all_files))  # virtual length

    def __getitem__(self, idx):
        # sample randomly to form a positive or negative pair
        import random
        if random.random() < 0.5:
            # positive pair (both genuine)
            if len(self.genuine) < 2:
                a = b = self.genuine[0]
            else:
                a, b = random.sample(self.genuine, 2)
            label = 1.0
        else:
            # negative pair (genuine vs forged)
            if not self.genuine or not self.forged:
                # fallback to random
                a = random.choice(self.all_files)
                b = random.choice(self.all_files)
                label = 1.0 if ('genuine' in a and 'genuine' in b) else 0.0
            else:
                a = random.choice(self.genuine)
                b = random.choice(self.forged)
                label = 0.0

        img1 = load_image(a)
        img2 = load_image(b)
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            tf = T.Compose([T.Resize((224, 224)), T.ToTensor()])
            img1 = tf(img1)
            img2 = tf(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


# ----------------------------- Seal Verifier ---------------------------

class SealVerifier:
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        self.device = device
        self.embedding_net = EmbeddingNet(embedding_dim=256)
        self.model = SiameseNet(self.embedding_net).to(self.device)
        self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                   std=[0.229, 0.224, 0.225])])
        if model_path and os.path.isfile(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def encode(self, pil_img: Image.Image) -> np.ndarray:
        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.embedding_net(x)
        return emb.cpu().numpy()[0]

    def verify(self, probe_img: Image.Image, reference_embs: List[np.ndarray], threshold: float = 0.6) -> Tuple[bool, float]:
        """Given a probe image of a seal and a list of reference embeddings (genuine seals), compute similarity
        Returns (is_match, min_distance)
        """
        e = self.encode(probe_img)
        dists = [np.linalg.norm(e - r) for r in reference_embs]
        min_d = float(np.min(dists)) if dists else float('inf')
        is_match = min_d < threshold
        return is_match, min_d

    def train(self, dataset_root: str, save_path: str, epochs: int = 10, batch_size: int = 16, lr: float = 1e-4):
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
        ds = SiamesePairDataset(dataset_root, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = ContrastiveLoss(margin=1.0)
        self.model.to(self.device)
        self.model.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (x1, x2, label) in enumerate(loader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                label = label.to(self.device)
                e1, e2 = self.model(x1, x2)
                loss = criterion(e1, e2, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item())
            print(f"Epoch {epoch+1}/{epochs} loss: {running_loss / (i+1):.4f}")
            torch.save(self.model.state_dict(), save_path)
        print('Training complete. Model saved to:', save_path)


# ----------------------------- Signature Verifier -----------------------

class SignatureVerifier(SealVerifier):
    """Identical structure to SealVerifier but could use different training data or augmentations.
    """
    pass


# ----------------------------- Text Inconsistency Detector -------------

class TextInconsistencyDetector:
    def __init__(self, model_path: Optional[str] = None):
        # a lightweight scikit-learn pipeline that takes feature vectors and outputs anomaly probability
        # features per text box are derived from ELA + geometric properties
        self.pipeline: Pipeline
        if model_path and os.path.isfile(model_path):
            self.pipeline = joblib.load(model_path)
        else:
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(n_estimators=100))
            ])

    def extract_features_from_image(self, pil_img: Image.Image) -> Tuple[np.ndarray, List[Tuple[str, Tuple[int, int, int, int]]]]:
        boxes = ocr_text_boxes(pil_img)
        feats = []
        for text, box in boxes:
            left, top, w, h = box
            f = ela_features_from_box(pil_img, box)
            # add geometric features
            area = (w * h) / (pil_img.width * pil_img.height + 1e-9)
            aspect = float(w) / (h + 1e-9)
            len_char = len(text)
            f_extended = np.concatenate([f, np.array([area, aspect, len_char], dtype=np.float32)])
            feats.append(f_extended)
        if feats:
            X = np.stack(feats, axis=0)
        else:
            X = np.zeros((0, 9), dtype=np.float32)
        return X, boxes

    def train(self, training_examples: List[Tuple[Image.Image, List[bool]]], save_path: Optional[str] = None):
        """
        training_examples: list of tuples (pil_img, labels_for_boxes)
        labels_for_boxes: list of booleans indicating whether each box is suspicious (True) or clean (False)
        Boxes must correspond to ocr_text_boxes order.
        """
        X_all = []
        y_all = []
        for pil_img, labels in training_examples:
            X, boxes = self.extract_features_from_image(pil_img)
            if len(X) == 0:
                continue
            if len(labels) != len(boxes):
                raise ValueError('labels must match number of OCR boxes for image')
            X_all.append(X)
            y_all.append(np.array([1 if b else 0 for b in labels], dtype=np.int32))
        X_stack = np.vstack(X_all)
        y_stack = np.concatenate(y_all)
        self.pipeline.fit(X_stack, y_stack)
        if save_path:
            joblib.dump(self.pipeline, save_path)
        print('Trained text-inconsistency detector on', X_stack.shape[0], 'boxes')

    def predict_image(self, pil_img: Image.Image, threshold: float = 0.5) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        X, boxes = self.extract_features_from_image(pil_img)
        results = []
        if X.shape[0] == 0:
            return results
        probs = self.pipeline.predict_proba(X)[:, 1]
        for (text, box), p in zip(boxes, probs):
            results.append((text, box, float(p)))
        return results


# ----------------------------- Demo / Usage -----------------------------

if __name__ == '__main__':
    # Simple demo scaffold. Replace paths with real images and trained models.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seal_model', type=str, default=None)
    parser.add_argument('--sig_model', type=str, default=None)
    parser.add_argument('--text_model', type=str, default=None)
    parser.add_argument('--reference_seal_dir', type=str, default='refs/seals')
    parser.add_argument('--probe', type=str, required=False)
    args = parser.parse_args()

    # Initialize modules
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seal_verifier = SealVerifier(model_path=args.seal_model, device=device)
    sig_verifier = SignatureVerifier(model_path=args.sig_model, device=device)
    text_detector = TextInconsistencyDetector(model_path=args.text_model)

    # load reference embeddings for seals (genuine seals)
    reference_embs = []
    if os.path.isdir(args.reference_seal_dir):
        for f in os.listdir(args.reference_seal_dir):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img = load_image(os.path.join(args.reference_seal_dir, f))
            reference_embs.append(seal_verifier.encode(img))

    if args.probe:
        probe_img = load_image(args.probe)
        # run OCR + text anomaly detection
        txt_results = text_detector.predict_image(probe_img)
        print('Text anomaly results (text, box, probability):')
        for t, box, p in txt_results:
            print(t, box, f'{p:.3f}')

        # assume probe contains a seal crop - in practice you'd detect the seal bounding box first
        is_match, dist = seal_verifier.verify(probe_img, reference_embs, threshold=0.6)
        print(f'Seal match: {is_match}, distance={dist:.4f}')

        # for signature verification you'd crop signature region, then call sig_verifier.verify
        # e.g. sig_crop = probe_img.crop((x,y,x+w,y+h))
    else:
        print('No probe image provided. This module is a library; use --probe <image> to run a simple demo.')

    print('Done')


"""
LINE-BY-LINE EXPLANATION (walkthrough)
=====================================

This appended block explains the Python module **Multimodal Forgery Detection** that appears above. I explain the code sequentially by logical block and indicate what each important line or small group of lines does. I avoid repeating full code lines verbatim (the code is visible above); instead I reference variable and function names so you can easily find each line in the file.

1) File header / module docstring
   - Explains the purpose of the file and lists dependencies and notes. Treat it as a high-level guide: what the module implements (seal/signature Siamese verifiers, ELA utilities, OCR-based text anomaly detector) and which external packages are required (PyTorch, torchvision, PIL, pytesseract, scikit-learn, OpenCV, etc.).

2) Standard library & third-party imports
   - `os, io, math`: filesystem, byte-streams, and math helpers.
   - `typing` import: used to annotate function return types (List, Tuple, Optional).
   - `numpy as np`: numerical arrays.
   - `PIL.Image, ImageChops`: image I/O and pixel-difference utilities used for ELA.
   - `cv2`: OpenCV for edge detection, Laplacian, and image operations.
   - `pytesseract`: OCR to extract text boxes.
   - `torch, torch.nn, DataLoader, Dataset, torchvision.transforms`: PyTorch model and data utilities.
   - `torchvision.models`: for loading a pretrained ResNet backbone.
   - `sklearn` imports and `joblib`: used to build, save, and load the RandomForest classifier pipeline for text-inconsistency detection.

3) Utilities (`ensure_rgb`, `load_image`)
   - `ensure_rgb(img)`: if the incoming PIL image is already RGB, return it; otherwise convert it to RGB. This avoids mode errors later when saving as JPEG or passing to transforms.
   - `load_image(path)`: opens an image from disk using PIL and ensures it is RGB via the helper above.

4) Error Level Analysis (ELA) functions
   - `error_level_analysis(pil_img, quality=90)`:
     * Ensures the image is RGB.
     * Saves the image to an in-memory JPEG (BytesIO) at the specified `quality` setting; this simulates recompression.
     * Re-opens the compressed image and computes the pixel-wise difference between the original and recompressed versions using `ImageChops.difference`.
     * Computes the maximum difference among channels (`getextrema`) and derives a `scale` factor so differences are visually boosted.
     * Attempts to use `PIL.ImageEnhance.Brightness` to multiply the difference image by `scale`. If that import fails, the code falls back to using `point(lambda p: p * scale)`.
     * Returns a scaled difference image (ELA image) — brighter pixels indicate higher recompression differences and thus possible tampering.
   - `ela_features_from_box(pil_img, box)`:
     * Accepts a box `(left, top, width, height)` and crops+resizes it to a standard shape (224x224) for stable features.
     * Calls `error_level_analysis` on the crop and converts to grayscale.
     * Converts the ELA image to a NumPy array in the range [0,1] and computes simple statistical descriptors: mean, std, max, min.
     * Computes an edge density using Canny edge detector and normalizes by the area of the crop.
     * Computes variance of Laplacian (a blur / resampling indicator).
     * Returns a small feature vector combining these descriptors. These features are intended for the text anomaly classifier.

5) OCR helpers (`ocr_text_boxes`)
   - Uses `pytesseract.image_to_data` with output_type=DICT to get word-level OCR results: text and bounding box coordinates.
   - Iterates over OCR results, skips empty text, and collects `(text, (left, top, width, height))` tuples.
   - These boxes are the input units for the ELA + geometric feature extraction.

6) Siamese network & embedding model
   - `EmbeddingNet` class:
     * Loads a pretrained ResNet18 backbone (optionally pretrained on ImageNet) and discards its final fully-connected layer.
     * Packs the remaining ResNet layers into a `self.backbone` `nn.Sequential`.
     * Defines a compact MLP `self.fc` that maps the backbone feature vector to an embedding of size `embedding_dim` (default 256): Linear -> ReLU -> Linear.
     * `forward(x)`: runs the image through the backbone, flattens the spatial dimensions, runs the MLP, and L2-normalizes the resulting embedding vector across the feature dimension. The normalization is useful so Euclidean/cosine distances are stable.
   - `SiameseNet` class:
     * Wraps the `EmbeddingNet` and, given two inputs, returns their embeddings `(e1, e2)` — suitable for contrastive or distance-based loss.
   - `ContrastiveLoss` class:
     * Implements a contrastive loss with a margin: for a similar pair (label=1) the loss encourages small squared distance; for dissimilar pairs (label=0) it penalizes pairs that are closer than the margin.
     * Implementation detail: distances are computed, and the dissimilar loss uses `clamp(margin - sqrt(dist), min=0.0).pow(2)` so only close negatives produce loss.
     * The final loss returns the mean over the batch.

7) Dataset: `SiamesePairDataset`
   - Initialization (`__init__`):
     * Expects a `root` directory containing two subfolders: `genuine/` and `forged/` with image files.
     * Scans both folders and builds `self.genuine` and `self.forged` lists.
   - `__len__`: returns a "virtual" length (the code returns `max(1000, len(self.all_files))`) so the dataset can be iterated for long training runs even with small datasets — this is a convenience but not strictly necessary.
   - `__getitem__(idx)`: randomly constructs a pair:
     * With 50% probability, samples a positive pair (two genuine images) and sets `label=1.0`.
     * Otherwise, samples a negative pair (genuine vs forged) with `label=0.0`.
     * Loads both images and applies the provided `transform` (or a default resize+ToTensor if none provided).
     * Returns `(img1_tensor, img2_tensor, label_tensor)`.
   - Note: This is a simple sampling strategy and can be improved (balanced sampling, hard negative mining, ensuring different sources, etc.).

8) Seal verifier (`SealVerifier`)
   - `__init__(model_path=None, device='cpu')`:
     * Creates an `EmbeddingNet` and wraps it in `SiameseNet`.
     * Prepares a standard image transform (Resize to 224, convert to Tensor, normalize with ImageNet means/stds).
     * If `model_path` points to a saved `state_dict`, it loads it and sets the model to evaluation mode.
   - `encode(pil_img)`:
     * Applies the transform to the PIL image, runs the embedding network and returns a NumPy embedding vector.
   - `verify(probe_img, reference_embs, threshold=0.6)`:
     * Encodes the probe image.
     * Computes L2 distances between the probe embedding and each embedding in `reference_embs`.
     * Returns `is_match` (True if the minimum distance is below `threshold`) and the `min_distance`. This allows threshold-tuning on validation data.
   - `train(dataset_root, save_path, epochs, batch_size, lr)`:
     * Constructs the `SiamesePairDataset` with transforms and a PyTorch `DataLoader`.
     * Sets up the optimizer (Adam) and `ContrastiveLoss`.
     * Runs a standard training loop across epochs and batches: forward, compute loss, backward, optimizer step.
     * Saves model weights each epoch to `save_path`.
     * Notes: the training loop uses `self.model` directly; in production you'd add validation, checkpointing, better logging, and possible mixed-precision.

9) Signature verifier (`SignatureVerifier`)
   - This class simply inherits `SealVerifier`. In practice you might change data augmentations, crop logic, or thresholding specifics for signatures versus stamps but the architecture is shared.

10) Text inconsistency detector (`TextInconsistencyDetector`)
    - `__init__(model_path=None)`:
      * Either loads an existing scikit-learn pipeline from `model_path` (using `joblib`) or constructs a pipeline consisting of a `StandardScaler` followed by a `RandomForestClassifier`.
      * The pipeline expects feature vectors derived from ELA and geometric descriptors for each OCR box.
    - `extract_features_from_image(pil_img)`:
      * Runs `ocr_text_boxes` to get a list of `(text, box)` items.
      * For each box, calls `ela_features_from_box` to get image-based descriptors.
      * Also computes geometric features: relative area (box area / image area), aspect ratio, and the length of the recognized text.
      * Concatenates features into a 1-D array per box. Stacks them into an `X` matrix or returns an empty array if no boxes were found.
    - `train(training_examples, save_path=None)`:
      * Expects `training_examples` to be a list of `(pil_img, labels_for_boxes)` where `labels_for_boxes` aligns with the order of OCR boxes returned by `ocr_text_boxes`.
      * Extracts features for all boxes across all training images, concatenates them into `X_stack` and label vector `y_stack`.
      * Fits the pipeline and optionally saves it to disk with `joblib.dump`.
      * Notes: labeling text boxes is manual and potentially time-consuming. You may want to label only suspicious/important boxes (grades, names) or use weak supervision.
    - `predict_image(pil_img, threshold=0.5)`:
      * Extracts features `X` and OCR boxes.
      * If no boxes were found, returns an empty list.
      * Calls `self.pipeline.predict_proba` to obtain the probability that each box is suspicious and returns `(text, box, probability)` tuples.

11) Demo / CLI (`if __name__ == '__main__':`)
    - Uses `argparse` to accept optional paths for seal/signature model weights, text model path, a reference seals directory, and a `--probe` image path.
    - Creates the `SealVerifier`, `SignatureVerifier`, and `TextInconsistencyDetector` instances. Picks `device='cuda'` if available.
    - Loads reference embeddings by encoding every image found under the `reference_seal_dir` so those embeddings can be used to compare the probe.
    - If a `--probe` image is provided:
      * Runs the text anomaly `predict_image` and prints results.
      * Runs `seal_verifier.verify` using the entire probe image (note: the demo assumes probe is already a crop of the seal; in a real pipeline you'd first localize the seal region) and prints whether the minimum distance to references is under the threshold.
      * Demonstrates how you could also crop signature areas and run `sig_verifier.verify`.
    - If no probe is given, prints a brief usage message.

12) Practical notes, caveats & improvements
    - ELA is sensitive to image format and recompression: results depend on JPEG quality and the original file format. Use consistent file handling and calibrate `quality` for your data.
    - The dataset sampler in `SiamesePairDataset` is simplistic — consider deterministic pairing, balanced batch composition, or mining hard negatives.
    - Contrastive loss here uses basic margin behavior. Triplet loss or a batch-hard mining strategy can give better embeddings for verification tasks.
    - The `verify` method uses a fixed L2 threshold; you should compute operating characteristics (ROC, FAR/FRR) on a validation set and choose a threshold that matches your risk tolerance.
    - OCR (`pytesseract`) may produce different box layouts depending on image DPI, fonts, and language. Consider tuning Tesseract config options or using a dedicated layout/receipt OCR model for documents.
    - For production, isolate pre-processing (deskew, denoise), and build an object detector (YOLO/Detic) to reliably find seal/signature regions before verification.
    - Persisting embeddings: you may want to store multiple reference embeddings per institution/year and use robust aggregation (median) for matching.

If you'd like any of the following next steps, I can perform them directly inside the same canvas file:
 - Insert inline `# comments` before each function and critical line (actual commented code inside the file) so the explanation is co-located with the code.
 - Add unit tests and a small synthetic dataset to exercise the pipeline locally.
 - Provide a shorter, chat-pasteable summary or specific line-by-line expansion for a particular function you care most about.

"""
