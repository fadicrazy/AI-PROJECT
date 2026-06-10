"""
Multimodal Breast Cancer Detection — Training Script
Architecture (per diagram):
  Ultrasound + Histopathology + Chest X-ray
  → Label Encoding & Data Augmentation (per modality)
  → Feature Extraction via EfficientNetB0 (pre-trained, per modality)
  → Feature Fusion (Concatenation)
  → Dense(512, ReLU + Dropout)
  → Dense(256, ReLU + Dropout)
  → Output: Sigmoid (Malignant/Benign)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
BATCH_SIZE = 32
DROPOUT    = 0.4
LR         = 1e-4
EPOCHS_P1  = 15   # Phase 1: frozen base
EPOCHS_P2  = 30   # Phase 2: fine-tune

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(PROJECT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DATA_ROOT  = '/content/datasets'
US_DIR     = os.path.join(DATA_ROOT, 'ultrasound')
HISTO_DIR  = os.path.join(DATA_ROOT, 'histopathology')
XRAY_DIR   = os.path.join(DATA_ROOT, 'chest_xray')


# ── Data Augmentation (Resize, Normalize, Flip, Rotate, Zoom) ─────────────────
def make_datagen(validation=False):
    if validation:
        return ImageDataGenerator(rescale=1./255)
    return ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        fill_mode='nearest'
    )

def make_generators(data_dir, dg_train, dg_val):
    kw = dict(
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['benign', 'malignant'],
        seed=42
    )
    train_gen = dg_train.flow_from_directory(os.path.join(data_dir, 'train'), **kw)
    val_gen   = dg_val.flow_from_directory(os.path.join(data_dir, 'val'),   **kw)
    return train_gen, val_gen


# ── Feature Extraction — EfficientNetB0 (Pre-trained on ImageNet) ─────────────
def build_extractor(modality_name: str):
    inp  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name=f'input_{modality_name}')
    base = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_tensor=inp,
        pooling='avg'       # → (batch, 1280)
    )
    base.trainable = False  # Freeze for Phase 1
    return Model(inputs=inp, outputs=base.output, name=f'efficientnet_{modality_name}')


# ── Full Multimodal Model ──────────────────────────────────────────────────────
def build_multimodal_model():
    i_us    = layers.Input((IMG_SIZE, IMG_SIZE, 3), name='input_ultrasound')
    i_histo = layers.Input((IMG_SIZE, IMG_SIZE, 3), name='input_histopathology')
    i_xray  = layers.Input((IMG_SIZE, IMG_SIZE, 3), name='input_chest_xray')

    e_us    = build_extractor('ultrasound')
    e_histo = build_extractor('histopathology')
    e_xray  = build_extractor('chest_xray')

    # Feature Fusion — Concatenation (3 × 1280 = 3840 dims)
    fused = layers.Concatenate(name='feature_fusion')(
        [e_us(i_us), e_histo(i_histo), e_xray(i_xray)]
    )

    # Dense Layer 512 — ReLU + Dropout
    x = layers.Dense(512, activation='relu', name='dense_512')(fused)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT, name='dropout_512')(x)

    # Dense Layer 256 — ReLU + Dropout
    x = layers.Dense(256, activation='relu', name='dense_256')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT, name='dropout_256')(x)

    # Output — Sigmoid (Malignant / Benign)
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    return Model(
        inputs=[i_us, i_histo, i_xray],
        outputs=output,
        name='BreastCancer_Multimodal'
    )


# ── Combined Generator (keeps 3 modalities in sync) ──────────────────────────
def combined_gen(g_us, g_histo, g_xray):
    while True:
        x1, y = next(g_us)
        x2, _ = next(g_histo)
        x3, _ = next(g_xray)
        yield [x1, x2, x3], y


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Breast Cancer Multimodal Training")
    print("  Ultrasound + Histopathology + Chest X-ray")
    print("  Feature Extractor: EfficientNetB0 (ImageNet)")
    print("=" * 60)

    dg_tr = make_datagen(validation=False)
    dg_vl = make_datagen(validation=True)

    tr_us,   vl_us   = make_generators(US_DIR,    dg_tr, dg_vl)
    tr_hs,   vl_hs   = make_generators(HISTO_DIR, dg_tr, dg_vl)
    tr_cx,   vl_cx   = make_generators(XRAY_DIR,  dg_tr, dg_vl)

    steps = min(tr_us.samples, tr_hs.samples, tr_cx.samples) // BATCH_SIZE
    vstep = min(vl_us.samples, vl_hs.samples, vl_cx.samples) // BATCH_SIZE
    print(f"Steps/epoch: {steps}  |  Val steps: {vstep}\n")

    model = build_multimodal_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.summary()

    cbs = [
        EarlyStopping(monitor='val_auc', patience=7,
                      restore_best_weights=True, mode='max', verbose=1),
        ModelCheckpoint(
            os.path.join(MODELS_DIR, 'multimodal_best.keras'),
            monitor='val_auc', save_best_only=True, mode='max', verbose=1
        ),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1)
    ]

    # Phase 1 — frozen base, train fusion + dense only
    print("\n[Phase 1] Training top layers (EfficientNetB0 frozen)...")
    model.fit(
        combined_gen(tr_us, tr_hs, tr_cx),
        steps_per_epoch=steps,
        validation_data=combined_gen(vl_us, vl_hs, vl_cx),
        validation_steps=vstep,
        epochs=EPOCHS_P1,
        callbacks=cbs
    )

    # Phase 2 — unfreeze top-20 layers of each extractor
    print("\n[Phase 2] Fine-tuning top-20 layers of each EfficientNetB0...")
    for lyr in model.layers:
        if lyr.name.startswith('efficientnet_'):
            lyr.trainable = True
            subs = lyr.layers if hasattr(lyr, 'layers') else []
            for i, s in enumerate(subs):
                s.trainable = (i >= len(subs) - 20)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR / 10),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    model.fit(
        combined_gen(tr_us, tr_hs, tr_cx),
        steps_per_epoch=steps,
        validation_data=combined_gen(vl_us, vl_hs, vl_cx),
        validation_steps=vstep,
        epochs=EPOCHS_P2,
        callbacks=cbs
    )

    # Evaluate
    print("\n[Evaluation] Running on validation set...")
    y_true, y_pred = [], []
    for _ in range(vstep):
        bx, by = next(combined_gen(vl_us, vl_hs, vl_cx))
        preds = model.predict(bx, verbose=0).flatten()
        y_pred.extend(preds.tolist())
        y_true.extend(by.tolist())

    y_labels = [1 if p >= 0.5 else 0 for p in y_pred]
    auc = roc_auc_score(y_true, y_pred)
    acc = sum(a == b for a, b in zip(y_true, y_labels)) / len(y_true)

    print(classification_report(y_true, y_labels, target_names=['Benign', 'Malignant']))
    print(f"ROC-AUC: {auc:.4f}")

    # Save
    model.save(os.path.join(MODELS_DIR, 'multimodal_model.keras'))
    print(f"\n✅ multimodal_model.keras saved to {MODELS_DIR}")

    metrics = {
        "model": "Multimodal EfficientNetB0 (US + Histo + CXR)",
        "val_accuracy": round(acc, 4),
        "roc_auc": round(auc, 4),
        "modalities": ["ultrasound", "histopathology", "chest_xray"],
        "architecture": {
            "feature_extractor": "EfficientNetB0 x3 (ImageNet pretrained)",
            "fusion": "Concatenation (3 x 1280 = 3840 dims)",
            "dense_layers": [512, 256],
            "activation": "ReLU + Dropout",
            "output": "Sigmoid (Malignant/Benign)"
        }
    }
    with open(os.path.join(PROJECT_DIR, 'multimodal_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ multimodal_metrics.json saved!")
    return model, metrics


if __name__ == '__main__':
    train()
