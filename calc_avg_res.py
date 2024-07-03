import os
import glob
import re

# Get list of all stdout files
# path = '/home/danielochana/MagNetTrafficFlowForecast/Output/midtertm2/1'
path = '/home/danielochana/MagNetTrafficFlowForecast/Output/spatial_2_classes_into_Nsym/1/'
files = glob.glob(os.path.join(path, '**', 'stdout'), recursive=True)

# Initialize sums and count
val_acc_latest_sum = 0
test_acc_latest_sum = 0
count = 0

val_avgs=[]
test_avgs = []
# Loop over each file
for file in files:
    with open(file, 'r') as f:
        contents = f.read()
        val_acc_latest_values = [float(x) for x in re.findall(r'val_acc_latest: (\d+\.\d+)', contents)]
        test_acc_latest_values = [float(x) for x in re.findall(r'test_acc_latest: (\d+\.\d+)', contents)]
        if val_acc_latest_values and test_acc_latest_values:
            val_avgs.append(sum(val_acc_latest_values) / len(val_acc_latest_values))
            test_avgs.append(sum(test_acc_latest_values) / len(test_acc_latest_values))
            val_acc_latest_sum += sum(val_acc_latest_values) / len(val_acc_latest_values)
            test_acc_latest_sum += sum(test_acc_latest_values) / len(test_acc_latest_values)
            count += 1

# Calculate averages
val_acc_latest_avg = val_acc_latest_sum / count if count else None
test_acc_latest_avg = test_acc_latest_sum / count if count else None

print(f'Average val_acc_latest: {val_acc_latest_avg}')
print(f'Average test_acc_latest: {test_acc_latest_avg}')
print(f'val_avgs: {val_avgs}')
print(f'test_avgs: {test_avgs}')
print(f'best val_acc_latest: {max(val_avgs)}')
print(f'best test_acc_latest: {max(test_avgs)}')