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
      plate_date = row[2]
      plate_info = parse_license_plate(id, plate_number)
      plate_info.date = plate_date
      if (train):
        price = row[3]
        plate_info.price = price
      # append to the list
      plate_info_list.append(plate_info)
  
  return plate_info_list

def region_code_sanity():
    plate_info_list = get_license_plate_info_list(train=True)
    # sanity check for regions
    regions = {}
    for plate_info in plate_info_list:
        regioncodes = regions.get(plate_info.region_name, [])
        if plate_info.region_code not in regioncodes:
            regioncodes.append(plate_info.region_code)
            regions[plate_info.region_name] = regioncodes

    print(f"Regions Count: {len(regions)}")

    for region_name in sorted(regions.keys()):
        regioncodes = regions[region_name]
        print(f"{region_name}: [", end="") 
        for regioncode in regioncodes:
            print(f"{regioncode}", end=", ")
        print("]")
  
def seeShapeOfPlateInfo(samples=20):
  # print each attribute of the plate_info object
  plate_info_list = get_license_plate_info_list(train=True)
  for plate_info in plate_info_list[:samples]:
    if(not plate_info.is_government_vehicle): continue
    print(f"\nplate_number: {plate_info.plate_number}")
    print(f"  id: {plate_info.id}")
    print(f"  region_Name: {plate_info.region_name}")
    print(f"  price: {plate_info.price}")
    print(f"  valid: {plate_info.valid}")
    print(f"  letter1: {plate_info.letter1}")
    print(f"  digits: {plate_info.digits}")
    print(f"  letter2: {plate_info.letter2}")
    print(f"  letter3: {plate_info.letter3}")
    print(f"  region_code: {plate_info.region_code}")
    print(f"  Error: {plate_info.error}")
    print(f"  is_government_vehicle: {plate_info._is_government_vehicle}")
    print(f"  government_info: ")
    print(f"    description: {plate_info.government_info['description']}")
    print(f"    forbidden_to_buy: {plate_info.government_info['forbidden_to_buy']}")
    print(f"    road_advantage: {plate_info.government_info['road_advantage']}")
    print(f"    significance_level: {plate_info.government_info['significance_level']}") 
    print(f"  date: {plate_info.date}")  

if __name__ == "__main__":
  seeShapeOfPlateInfo(10_000)
  
'''plate_number: K585OO77
  id: 9989
  region_Name: Moscow
  price: 1250000
  valid: True
  letter1: K
  digits: 585
  letter2: O
  letter3: O
  region_code: 77
  Error: None
  is_government_vehicle: True
  government_info:
    description: Partially Constitutional Court plates
    forbidden_to_buy: False
    road_advantage: True
    significance_level: 3'''
   
    
    # if(plate_info.government_info):
    #   print(f"\nPlate Number: {plate_info.plate_number}, Region: {plate_info.region_name}, Price: {plate_info.price}")
    #   print(f"  Government Info: {plate_info.government_info['description']}")
    #   print(f"  Forbidden to buy: {plate_info.government_info['forbidden_to_buy']}")
    #   print(f"  Road advantage: {plate_info.government_info['road_advantage']}")
    #   print(f"  Significance level: {plate_info.government_info['significance_level']}")
    

    
