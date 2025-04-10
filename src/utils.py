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
  '''{
      'valid':
      'letter1':
      'digits':
      'letter2':
      'letter3':
      'region_code':
      'region_name':
      'government_info':
      'error':
      'price':
      'plate_number':
      'id':
        }'''
  
  '''plate_info._government_info = {
                'description': string,
                'forbidden_to_buy': bool,
                'road_advantage': bool,
                'significance_level': string
            }'''
  def region_code_sanity():
    plate_info_list = get_license_plate_info_list(train=True)
    # sanity check for regions
    regions = {}
    for plate_info in plate_info_list:
      regioncodes = regions.get(plate_info.region_name, [])
      if(plate_info.region_code not in regioncodes):
        regioncodes.append(plate_info.region_code)
        regions[plate_info.region_name] = regioncodes
      # print(f"{plate_info.region_code}, {plate_info.region_name}")
    print(f"Regions Count: {len(regions)}")
    for region_name, regioncodes in regions.items():
      print(f"{region_name}: [", end="") 
      for regioncode in regioncodes:
        print(f"{regioncode}",end=", ")
      print("]")
    
    # if(plate_info.government_info):
    #   print(f"\nPlate Number: {plate_info.plate_number}, Region: {plate_info.region_name}, Price: {plate_info.price}")
    #   print(f"  Government Info: {plate_info.government_info['description']}")
    #   print(f"  Forbidden to buy: {plate_info.government_info['forbidden_to_buy']}")
    #   print(f"  Road advantage: {plate_info.government_info['road_advantage']}")
    #   print(f"  Significance level: {plate_info.government_info['significance_level']}")
    

    
