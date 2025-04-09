from parse_plate import *
import csv
# Open the train CSV file

with open('data/train.csv', mode='r', encoding='utf-8') as file:
  reader = csv.reader(file)
  next(reader)  # Skip the header row if it exists

  for row in reader:
    plate_number = row[1] 
    plate_info = parse_license_plate(plate_number)
    price = row[3]
    
    if plate_info.valid:
      print(f"Plate Number: {plate_number}, Region: {plate_info.region_name}, Price: {price}")
# read the train csv in
# for each line in the csv
    # parse the plate number
    # if the plate number is valid
        # print the plate number and the region name