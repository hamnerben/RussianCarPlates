from parse_plate import *
import csv
# Open the train CSV file

def get_license_plate_info_list(train=True):
  plate_info_list = []
  file_path = 'src/data/train.csv' if train else 'src/data/test.csv'
  with open(file_path, mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row if it exists

    for row in reader:
      # create the plate_info object
      id = row[0]
      plate_number = row[1] 
      plate_info = parse_license_plate(id, plate_number)
      if(train):
        price = row[3]
        plate_info.price = price
      # append to the list
      plate_info_list.append(plate_info)
  
  return plate_info_list


if __name__ == "__main__":
  plate_info_list = get_license_plate_info_list(train=True)
  for plate_info in plate_info_list:
    print(f"Plate Number: {plate_info.plate_number}, Region: {plate_info.region_name}, Price: {plate_info.price}")
    
