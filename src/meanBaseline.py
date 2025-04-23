from parse_plate import *
import csv

# Open the train CSV file
with open('src/data/train.csv', mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row if it exists

    # Calculate the total price and count the rows
    totals_price = 0
    row_count = 0
    for row in reader:
        price = float(row[3]) 
        totals_price += price
        row_count += 1

    mean_price = totals_price / row_count  # Calculate mean price

# Open the test CSV file to get the IDs
with open('src/data/test.csv', mode='r', encoding='utf-8') as test_file:
    test_reader = csv.reader(test_file)
    test_header = next(test_reader)  # Skip the header row if it exists

    # Write the mean price for every ID from test.csv to submission.csv
    with open('src/data/mean_baseline.csv', mode='w', encoding='utf-8', newline='') as submission_file:
        writer = csv.writer(submission_file)
        writer.writerow(['id', 'price'])  

        for test_row in test_reader:
            plate_id = test_row[0] 
            writer.writerow([plate_id, mean_price]) 
