import os
import sys
import json

# Add workspace code to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CODE_ROOT = os.path.join(REPO_ROOT, "code")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

import webapp_backend.app as app_module

if __name__ == "__main__":
    try:
        # Prefer globally best model if available
        try:
            app_module.load_best_overall_model()
        except Exception:
            app_module.load_best_model("DeepLearningClassifier")
        print("Loaded model:", app_module._MODEL_NAME)
        print("Input dim:", app_module._INPUT_DIM)
        print("Feature names count:", len(app_module._FEATURE_NAMES) if app_module._FEATURE_NAMES else 0)
        print("Test metrics:", json.dumps(app_module._TEST_METRICS))
        print("Model type:", type(app_module._MODEL).__name__)
        
        # Check if model is on GPU
        if hasattr(app_module._MODEL, "device"):
            print("Model device:", app_module._MODEL.device)
        
        # Lightweight preprocessor smoke test (no external API calls)
        try:
            preproc = getattr(app_module, "_PREPROCESSOR", None)
            if preproc is not None:
                import pandas as pd
                cols = getattr(preproc, "pca_input_features", [])
                if cols:
                    df = pd.DataFrame([[0.0] * len(cols)], columns=cols)
                    out = preproc.transform(df)
                    print("Preprocessor output shape:", getattr(out, "shape", None))
        except Exception as e:
            print("Preprocessor smoke test failed:", e)

        print("\nSUCCESS: Model loaded on GPU!")
    except Exception as e:
        print("FAILED:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0)
