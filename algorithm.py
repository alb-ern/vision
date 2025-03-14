import numpy as np
import thing
import matplotlib.pyplot as plt
import tqdm


# CODE


lr = 0.05
images = thing.images
labels = thing.labels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigma(x):
    return sigmoid(x * 4)

# image turned into 1d normalized array


def take_image(num):
    image = images[num]
    normal_image = image / 128 - 1
    input_layer = normal_image.reshape(1, 784)
    return input_layer


# functions applied
class Agent:
    agents = []

    def __init__(self, real: bool = True) -> None:
        self.w1 = np.random.rand(784, 900) * 2 - 1
        self.b1 = np.random.rand(900) * 2 - 1
        self.w2 = np.random.rand(900, 300) * 2 - 1
        self.b2 = np.random.rand(300) * 2 - 1
        self.w3 = np.random.rand(300, 10) * 2 - 1
        self.b3 = np.random.rand(10) * 2 - 1
        self.params = [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3]
        if real:
            Agent.agents.append(self)

    def see(self, num):
        input_layer = take_image(num)
        l2 = np.maximum(0, (input_layer.dot(self.w1) / 100 + self.b1))
        l3 = sigmoid(l2.dot(self.w2) / 100 + self.b2)
        output_layer = sigmoid(l3.dot(self.w3) / 50 + self.b3)
        out_sum = np.sum(output_layer)
        normalized_out = output_layer / out_sum
        return normalized_out

    def success(self, num):
        answer = self.see(num)
        correct_answer = labels[num]
        return answer[0, correct_answer]


for i in range(50):
    a = Agent()

for i in range(100):
    for ag in Agent.agents:
        ag.current_success = ag.success(0)
    Agent.agents = sorted(Agent.agents, key=lambda obj: obj.current_success, reverse=True)[:25]
    new_gen = []
    for ag in Agent.agents:
        new = Agent(real=False)
        for ix in range(6):
            new.params[ix] = ag.params[ix] * (1 - lr) + np.random.rand(*ag.params[ix].shape) * lr
        new_gen.append(new)
    Agent.agents.extend(new_gen)
    print(Agent.agents[0].current_success)




# a0 = Agent()
# a1 = Agent()
# success0 = a0.success(0)
# success1 = a1.success(0)
# for i in tqdm.tqdm(range(1, 1000)):
#     ix,r = np.random.choice(range(5),2)

#     rel_success = success1 - success0
#     if rel_success > 0:
#         err = a1.params[ix] - a0.params[ix]
#         a1.params[ix] = a1.params[ix] + err * lr
#         a0.params[ix] = a0.params[ix]+err*lr*3
#     else:
#         err = a0.params[ix] - a1.params[ix]
#         a0.params[ix] = a0.params[ix] + err * lr
#         a1.params[ix] = a1.params[ix] + err * lr * 3
#     a0.params[r]=a0.params[r]+Agent().params[r]*lr
#     a1.params[r]=a1.params[r]+Agent().params[r]*lr
#     success0=a0.success(i)
#     success1=a1.success(i)
#     if i%100==0:
#         print(max(success0,success1))


# plt.imshow(test_image2, cmap='gray')
# plt.step(range(0, 10), out.reshape(10,))


plt.show()
