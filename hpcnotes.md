squeue -u $USER
sacct -u $USER --starttime today


# Navigate to the analysis directory first
cd /Users/nathanbehailu/Desktop/Capstone/capstone/analysis

# Activate your virtual environment
source ../venv/bin/activate

# Run the analysis
python -m src.analyze analyze ../results/mutag_fixed_test_seed_0.txt --outdir results --save-csv
