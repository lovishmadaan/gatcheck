import numpy as np
count=0
# fold_change=False
current_valid_max=0
valid_fold=[0,0,0,0,0]
test_fold=[0,0,0,0,0]
current_fold_number=0
for line in open("brightkite_link_go.txt"):
	if "Test AUC" in line:
		count=count+1
		line1=line.rstrip().split()
		if (float(line1[9])>current_valid_max):
			# print(float(line1[9]))
			# print(valid_fold)
			# print(current_fold_number)
			valid_fold[current_fold_number]=float(line1[9])
			test_fold[current_fold_number]=float(line1[12])
			current_valid_max=float(line1[9])
		if count==41:
			current_valid_max=0
			current_fold_number=current_fold_number+1
			count=0
print(valid_fold)
print(test_fold)
print(np.mean(valid_fold))
print(np.std(valid_fold))
print(np.mean(test_fold))
print(np.std(test_fold))