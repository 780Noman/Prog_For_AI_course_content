#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      DELL
#
# Created:     17/10/2023
# Copyright:   (c) DELL 2023
# Licence:     <your licence>
#-------------------------------------------------------------------------------

def main():
    pass

if __name__ == '__main__':
    main()

import datetime

prev_units = 0.00

def calculate_unit_price(units_consumed):
    if 0 <= units_consumed <= 100:
        return 13
    elif 101 <= units_consumed <= 200:
        return 18
    elif 201 <= units_consumed <= 300:
        return 22
    elif 301 <= units_consumed <= 400:
        return 25
    elif 401 <= units_consumed <= 500:
        return 27
    elif 501 <= units_consumed <= 600:
        return 29
    elif 601 <= units_consumed <= 700:
        return 30
    else:
        return 35

def calculate_gst(total_cost):
    return (total_cost / 100) * 17

def calculate_nj_surcharge(units_consumed):
    return (units_consumed / 100) * 10

def calculate_fc_surcharge(units_consumed):
    return (units_consumed / 100) * 43

def calculate_fuel_adjustment(units_consumed):
    return (units_consumed / 100) * 20.22

while True:
    user = int(input("Press 1 to continue: "))
    if user != 1:
        break

    customer_name = input("Enter customer name: ")
    units_consumed = int(input("Enter total number of units consumed: "))
    date_str = input("Enter due date (YYYY-MM-DD): ")
    expire_date_str = input("Enter expire date (YYYY-MM-DD): ")

    total_electricity_cost = 0.00
    tv_fee = 35.00

    unit_price = calculate_unit_price(units_consumed)

    if prev_units > 300:
        print("Previous month units were greater than 300")
        unit_price *= 25
    else:
        print("Previous month units were less than 300")
        unit_price = calculate_unit_price(units_consumed)

    total_electricity_cost = units_consumed * unit_price
    gst = calculate_gst(total_electricity_cost)
    nj_surcharge = calculate_nj_surcharge(units_consumed)
    fc_surcharge = calculate_fc_surcharge(units_consumed)
    fuel_adjustment = calculate_fuel_adjustment(units_consumed)

    total_electricity_cost += tv_fee + gst + fuel_adjustment + fc_surcharge + nj_surcharge

    date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    expire_date = datetime.datetime.strptime(expire_date_str, "%Y-%m-%d")

    if date < expire_date:
        print("**********************")
        print("Paying before due date with 0% fine")
        result = total_electricity_cost
    else:
        print("**********************")
        print("Paying after due date with 2% fine")
        result = total_electricity_cost
        fine = (result / 100) * 2
        result += fine

    print("Customer name: ", customer_name)
    print("Units Consumed: ", units_consumed)
    print("TV Fee: ", tv_fee)
    prev_units = units_consumed
    print("NJ Surcharge: ", nj_surcharge)
    print("FC Surcharge: ", fc_surcharge)
    print("Fuel Adjustment: ", fuel_adjustment)
    print("Total GST: ", gst)
    print("Total Bill: ", result)