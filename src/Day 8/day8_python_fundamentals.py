#Task-1
import numpy as np
scores = np.random.randint(50, 101, size=(5, 3))
subject_mean = scores.mean(axis=0)
centered_scores = scores - subject_mean
print("Original Scores:\n", scores)
print("Subject-wise Mean:\n", subject_mean)
print("Centered Scores:\n", centered_scores)

#Task-2
import numpy as np
data=np.arange(24)
reshaped=data.reshape(4,3,2)
transposed=reshaped.transpose(1,0,2)
print("Final shape:\n",transposed.shape)
print("Final array:\n",transposed)
