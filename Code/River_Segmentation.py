import numpy as np
import matplotlib.pyplot as plt

river_name = 'jw'
hand_sum_list = np.loadtxt('HAND_sum_list_'+river_name+'.txt')
han_count_list = np.loadtxt('HAN_count_list_'+river_name+'.txt')
segment_list = np.loadtxt('segment_'+river_name+'.txt')

average_hand = list(hand_sum_list / han_count_list)
segment_list = list(map(int, segment_list))
change_list = [abs(average_hand[i] - average_hand[i+1])/average_hand[i] for i in range(len(average_hand)-1)]
print(change_list)
target_list = []
for i in range(len(change_list)):
    if change_list[i] <= 1:
        target_list.append(i)
print(target_list)
for idx, value in enumerate(target_list):
    if idx != 0 and target_list[idx] - target_list[idx-1] == 1:
        average_hand[value + 1] = sum(hand_sum_list[value-1:value + 2]) / sum(han_count_list[value-1:value + 2])
        hand_sum_list[value + 1] = sum(hand_sum_list[value-1:value + 2])
        han_count_list[value + 1] = sum(han_count_list[value-1:value + 2])
        segment_list[value + 1] = 0
        segment_list[value] = 0
    else:
        average_hand[value+1] = sum(hand_sum_list[value:value+2]) / sum(han_count_list[value:value+2])
        hand_sum_list[value + 1] = sum(hand_sum_list[value:value + 2])
        han_count_list[value + 1] = sum(han_count_list[value:value + 2])
        segment_list[value + 1] = 0
    average_hand[value] = 0
    hand_sum_list[value] = 0
    han_count_list[value] = 0
hand_sum_list = list(hand_sum_list)
han_count_list = list(han_count_list)
print(len(average_hand))
while 0 in average_hand:
    average_hand.remove(0)
    segment_list.remove(0)
    hand_sum_list.remove(0)
    han_count_list.remove(0)

print(average_hand)
print(segment_list)
print(len(segment_list))
# plt.plot(average_hand, marker='o')
# plt.show()
#
# change_list = [abs(average_hand[i] - average_hand[i+1])/average_hand[i] for i in range(len(average_hand)-1)]
# for i in change_list:
#     if i <= 0.2:
#         print('*****')
