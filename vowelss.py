#-------------------------------------------------------------------------------
# Name:        module1
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
user_input = input("Enter a name: ")

count = 0
char=['a','e','i','o','u']
for i in user_input:
    if i.lower() in char:
        count = count + 1
        print(i)
print(count)