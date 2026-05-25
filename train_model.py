import os, sys
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
import financial_advisor_combined_ml as fa
fa.train("data.csv")
print("TRAINING COMPLETE")
