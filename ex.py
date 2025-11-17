import math
import numpy as np

# output = [0.95, 0, 0]
target = [1, 0, 0]

# # loss = -1 * (math.log(output[0]) * target[0] + 
# #              math.log(output[1]) * target[1] + 
# #              math.log(output[2]) * target[2])


# loss = -1 * (math.log(output[0]))
# # loss = -log(output[0])
# print(loss)



# output = np.array([[0.7, 0.1, 0.2],
#           [0.1, 0.5, 0.4],
#           [0.02, 0.9, 0.08]])


# #[0, 0, 1]
# print(-np.log(1.000000001))


# targets = np.array([[1, 0, 0],
#                    [0, 1, 0],
#                    [0, 1, 0]])
# if len(targets.shape) == 1:
#     cc = output[
# range(len(output)),
# targets
# ]
# elif len(targets.shape) == 2:
#     cc = np.sum(
# output * targets,
# axis=1
# )
# avg = np.mean(-np.log(cc))
# print(avg)
# preds = np.argmax(outputs, axis = 1)

# accuracy = np.mean(predicitions)




outputs = np.array([[0.7, 0.1, 0.1],
                    [0.1, 0.5, 0.4],
                    [0.02, 0.9, 0.08]])

targets = np.array([0, 1, 1])

preds = np.argmax(outputs, axis = 1)
if len(targets.shape) == 2:
    targets = np.argmax(targets, axis=1)
    
accuracy = np.mean(preds == targets)
print(accuracy)