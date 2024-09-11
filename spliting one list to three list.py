#-------------------------------------------------------------------------------
# Name:        module2
# Purpose:
#
# Author:      DELL
#
# Created:     11/10/2023
# Copyright:   (c) DELL 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()
#Take 10 input from the user and divide input list index values equally to 3 new lists

inputs = []

for i in range(10):
    user_input = input(f"Enter input {i + 1}: ")
    inputs.append(user_input)

print("Inputs:", inputs)

my_list = inputs.copy()

c1 = my_list[0:3]
c2 = my_list[3:7]
c3 = my_list[6:10]

print("c1:", c1)
print("c2:", c2)
print("c3",c3)
