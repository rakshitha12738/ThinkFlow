# ThinkFlow — Emotion Prediction & Circadian Intelligence Engine

ThinkFlow is an adaptive developer companion and circadian intelligence platform that estimates developer emotional state and cognitive fatigue from typing dynamics and webcam affective signals, delivering non-intrusive flow nudges and break recommendations.

---

## 1. Production Emotion-Prediction Pipeline

ThinkFlow evaluates emotional signals along three operational states:
- **`frustration`**: Elevated backspace ratio, high error correction frequency, irregular pauses.
- **`confidence`**: High steady typing pace, low backspace ratio, smooth flow.
- **`neutral`**: Baseline typing behavior or mixed signals falling within the operational indifference band.

### Inference Flow
1. **Typing Telemetry Collection**: VS Code extension accumulates window metrics (characters typed, words typed, backspaces, error corrections, pause duration, window duration).
2. **Feature Extraction**: Extracted into a 10-feature vector using strictly bounded normalized ratios (`typingSpeedWpm`, `backspaceRatio`, `errorRate`, `pauseRatio`) and window counters.
3. **Primary ML Inference**: Executed in-process via `XGBoostEmotionClassifier` in `backend/src/services/xgbEmotionClassifier.ts`.
4. **Ensemble Softmax**: Gradient boosted decision tree margins are accumulated across classes and transformed via numerically stable softmax into calibrated probabilities.
5. **Deterministic Fallback**: If the model is not loaded, corrupt, or throws an unhandled condition, execution gracefully falls back to the deterministic rule-based engine in `emotionEngine.ts`.
6. **VibeScore Synthesis**: Maps continuous probabilities into ThinkFlow's standard `vibeScore` scale (0–100) and feeds adaptive companion mode selection (`gentle`, `balanced`, `challenge`).

---

## 2. Model Loading & In-Process Inference

The classifier is pure TypeScript/Node.js, evaluating JSON tree structures directly without requiring an external Python runtime or native C++ addons:
```typescript
import { XGBoostEmotionClassifier } from "./services/xgbEmotionClassifier.js";

const classifier = new XGBoostEmotionClassifier("models/xgboost-emotion.json");
const result = classifier.predictWithFallback({
  charsTyped: 300,
  wordsTyped: 60,
  backspaces: 4,
  errorCount: 1,
  pauseMs: 4000,
  durationMs: 60000,
});
```

- **Production Model**: `backend/models/xgboost-emotion.json` (v1.1.0)
- **Candidate Model**: `backend/models/xgboost-emotion-candidate.json` (v1.2.0-candidate)
- **Verified Rollback**: `backend/models/xgboost-emotion.json.bak`

---

## 3. Dataset Provenance & Label Definitions

### Academic Dataset
- **Reference**: Kołakowska & Landowska (2021), *Keystroke Dynamics Patterns While Writing Positive and Negative Opinions*, Sensors 2021.
- **Participants**: 49 independent subjects.
- **Samples**: 245 total rows (5 rows per participant).
  - Row index `0 mod 5`: Full-session aggregate vector (excluded from training to prevent label corruption).
  - Row indices `1–4 mod 5`: 4 opinion writing tasks (Task 1: Best teacher, Task 2: Worst teacher, Task 3: Best subject, Task 4: Worst subject).
- **Labels**: Continuous Pleasure ($P$), Arousal ($A$), Dominance ($D$) ratings collected on a 1–9 Self-Assessment Manikin (SAM) scale.

### Scientific Ground-Truth Clarification
- PAD ratings represent subjective dimensional affect, **not direct measures of developer IDE confidence or frustration**.
- Task polarity (writing positive vs negative opinions) serves as an experimental proxy for emotional valence.
- The dataset contains **zero ground-truth neutral samples**. The neutral class represents an operational indifference band and heuristic fallback.
- True 95%+ emotion recognition in real-world software engineering requires multimodal fusion (webcam facial affect + typing) and personalized baseline calibration.

---

## 4. Reproducible Training Pipeline

The Python training implementation is located in `backend/scripts/train_xgboost.py`.

### Run Training Pipeline (Windows PowerShell)
```powershell
# Using Anaconda or Python 3.12+ environment
C:\Users\Ramu\anaconda3\python.exe backend/scripts/train_xgboost.py
```

Options:
- `--data <path>`: Path to feature dataset CSV (defaults to benchmark generator if absent).
- `--labels <path>`: Path to PAD labels CSV (defaults to benchmark generator if absent).
- `--output <path>`: Target JSON export path (defaults to `backend/models/xgboost-emotion-candidate.json`).

---

## 5. Evaluation Protocol & Leakage Protection

The training pipeline enforces strict safeguards against data leakage and overfitting:
1. **Grouped Cross-Validation**: 5-Fold `GroupKFold` grouped strictly by participant ID. All 4 writing tasks from any participant remain exclusively in either the train or validation fold.
2. **Held-Out Test Set**: 10 unseen participants (participants 39–48, 40 samples) strictly held out from hyperparameter selection.
3. **Fold-Local Preprocessing**: Scalers and transformations are fitted strictly within training splits.
4. **Rate-Only Feature Prioritization**: Prioritizes duration-invariant rate metrics (`typingSpeedWpm`, `backspaceRatio`, `errorRate`, `pauseRatio`) over duration-dependent raw counts to eliminate window-length bias.

### Evaluated Model Comparison (5-Fold Grouped CV)
| Model | Grouped CV Accuracy | CV Std | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **Linear SVM** | 76.88% | ±0.0919 | 0.7666 |
| **Logistic Regression** | 76.88% | ±0.0919 | 0.7673 |
| **Random Forest** | 75.27% | ±0.0882 | 0.7463 |
| **XGBoost (Selected)** | **71.61%** | **±0.0623** | **0.7075** |

### Held-Out Unseen Participant Performance (XGBoost)
- **Baseline Accuracy**: 50.00% (random guess / majority class)
- **Training Accuracy**: 80.13%
- **Held-Out Test Accuracy**: 72.50%
- **Balanced Accuracy**: 72.50%
- **Macro-F1**: 0.7234
- **Per-Class Precision**: Frustration = 76.47%, Confidence = 69.57%
- **Per-Class Recall**: Frustration = 65.00%, Confidence = 80.00%
- **Inference Latency**: ~36.6 microseconds/sample
- **Model Size**: ~47.8 KB

---

## 6. Continuous PAD Dimensional Analysis

Evaluating continuous regression on Pleasure ($P$), Arousal ($A$), and Dominance ($D$) via Ridge regression:
- **Pleasure ($P$)**: $R^2 = 0.3254$, $\text{MAE} = 1.2133$
- **Arousal ($A$)**: $R^2 = -0.0080$, $\text{MAE} = 1.0949$
- **Dominance ($D$)**: $R^2 = 0.2036$, $\text{MAE} = 1.3036$

*Assessment*: Keystroke timing correlates moderately with Pleasure/Valence and Dominance, but weakly with Arousal when isolated from facial/physiological signals.

---

## 7. Model Promotion & Rollback

### Promotion Criteria
A candidate model is eligible for promotion only when:
1. Feature schema strictly matches `TYPING_FEATURE_NAMES` (10 features).
2. Held-out participant accuracy and macro-F1 exceed baseline without participant leakage.
3. In-process JSON parser validation passes.
4. Production prediction test suite passes.

### Rollback Procedure
If model regression occurs in production:
```powershell
# Restore backed up production artifact
Copy-Item backend/models/xgboost-emotion.json.bak backend/models/xgboost-emotion.json -Force
```

---

## 8. Privacy-Preserving Research Pipeline

ThinkFlow includes an opt-in research pipeline:
- **Disabled by default**: Explicit voluntary consent required (`optedIn: true`).
- **Pseudonymization**: User IDs are hashed using HMAC-SHA256 with a 32+ character server secret (`THINKFLOW_RESEARCH_SALT`).
- **Zero Raw Keystrokes**: Collects only aggregate numerical rates and counts; raw text, code, and passwords are never captured.
- **Right to Erasure (GDPR Art. 17)**: Users can withdraw consent and delete all contributed research samples via `DELETE /api/v1/users/:userId/research-data` or through the extension command `ThinkFlow: Manage Research Consent`.

---

## 9. Environment Variables

Create `.env` in `backend/` using `backend/.env.example`:
```ini
PORT=8080
HOST=127.0.0.1
FRONTEND_URL=http://localhost:5173
THINKFLOW_DB_PATH=./data/thinkflow-db.json
THINKFLOW_MODEL_PATH=./models/xgboost-emotion.json
THINKFLOW_RESEARCH_SALT=development_salt_at_least_32_characters_long_for_participant_pseudonymization
```

---

## 10. Development & Verification Guide

### Backend
```powershell
cd backend
npm install
npm run check    # TypeScript type check
npm run build    # Compile to dist/
npm test         # Run 29 comprehensive tests
npm start        # Start Express server on http://127.0.0.1:8080
```

### Extension
```powershell
cd extension
npm install
npm run build    # Compile VS Code extension to dist/extension.js
```

### Frontend
```powershell
cd frontend
npm install
npm run build    # Compile Vite frontend
npm run dev      # Run Vite dev server on http://localhost:5173
```
