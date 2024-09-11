#-------------------------------------------------------------------------------
# Name:        module4
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
# Initialize variables
total = 0
count = 0
smallest = float('inf')
largest = float('-inf')
num = int(input("Enter a positive integer (or 0 to stop): "))

# Read integers from the keyboard
while(num!=0 or num>0):
    # Check if input is 0 to stop the loop
    if num == 0:
        break

    # Update variables
    total += num
    count += 1
    smallest = min(smallest, num)
    largest = max(largest, num)

# Calculate average
average = total / count

# Calculate inclusive range
inclusive_range = largest - smallest + 1

# Print the results
print(f"Average value: {average}")
print(f"Smallest value: {smallest}")
print(f"Largest value: {largest}")
print(f"Inclusive range: {inclusive_range}")
##small=0;
##sm=0;
##count=0
##large=0;
##avg=0.0
##inclusive_range=0
##n=int(input("Please enter a number strictly positive :"))
##while(n!=0 or n>0):
##    count=count+1;
##    sm=sm+n
##    if n<small:
##        small=n
##    elif n>large:
##        large=n
##    n=int(input("Please enter a number strictly positive :")
##
##avg=sm/count
##print(f"The avg value of entered nums is {avg} ")
##print(f"The smallest value of entered nums is {small} ")
##print(f"The largest value of entered nums is {large} ")
##print(f"The inclusive range value of entered nums is {large-small+1} ")