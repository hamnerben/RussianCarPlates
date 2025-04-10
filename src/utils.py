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
      print(f"\nPlate Number: {plate_info.plate_number}")
      print(f"  ID: {plate_info.id}")
      print(f"  Region Name: {plate_info.region_name}")
      print(f"  Price: {plate_info.price}")
      print(f"  Valid: {plate_info.valid}")
      print(f"  Letter1: {plate_info.letter1}")
      print(f"  Digits: {plate_info.digits}")
      print(f"  Letter2: {plate_info.letter2}")
      print(f"  Letter3: {plate_info.letter3}")
      print(f"  Region Code: {plate_info.region_code}")
      print(f"  Error: {plate_info.error}")
      print(f"  Government Info: ")
      print(f"    Description: {plate_info.government_info['description']}")
      print(f"    Forbidden to buy: {plate_info.government_info['forbidden_to_buy']}")
      print(f"    Road advantage: {plate_info.government_info['road_advantage']}")
      print(f"    Significance level: {plate_info.government_info['significance_level']}")   

if __name__ == "__main__":
  seeShapeOfPlateInfo()
  
  '''Plate Number: P141BY77
  ID: 17
  Region Name: Moscow
  Price: 300000
  Valid: True
  Letter1: P
  Digits: 141
  Letter2: B
  Letter3: Y
  Region Code: 77
  Error: None
  Government Info: 
    Description: Non-government vehicle
    Forbidden to buy: False
    Road advantage: False
    Significance level: 0'''
   
    
    # if(plate_info.government_info):
    #   print(f"\nPlate Number: {plate_info.plate_number}, Region: {plate_info.region_name}, Price: {plate_info.price}")
    #   print(f"  Government Info: {plate_info.government_info['description']}")
    #   print(f"  Forbidden to buy: {plate_info.government_info['forbidden_to_buy']}")
    #   print(f"  Road advantage: {plate_info.government_info['road_advantage']}")
    #   print(f"  Significance level: {plate_info.government_info['significance_level']}")
    

    
