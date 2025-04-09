from supplemental_english import REGION_CODES, GOVERNMENT_CODES
import re

class LicensePlateInfo:
    def __init__(self):
        self._valid = False
        self._letter1 = None
        self._digits = None
        self._letter2 = None
        self._letter3 = None
        self._region_code = None
        self._region_name = None
        self._government_info = None
        self._error = None

    @property
    def valid(self): return self._valid

    @property
    def letter1(self): return self._letter1

    @property
    def digits(self): return self._digits

    @property
    def letter2(self): return self._letter2

    @property
    def letter3(self): return self._letter3

    @property
    def region_code(self): return self._region_code

    @property
    def region_name(self): return self._region_name

    @property
    def government_info(self): return self._government_info

    @property
    def error(self): return self._error

    def to_dict(self):
        return {
            'valid': self._valid,
            'letter1': self._letter1,
            'digits': self._digits,
            'letter2': self._letter2,
            'letter3': self._letter3,
            'region_code': self._region_code,
            'region_name': self._region_name,
            'government_info': self._government_info,
            'error': self._error
        }

def parse_license_plate(plate_number):
    VALID_LETTERS = set('ABEKMHOPCTYX')
    plate_info = LicensePlateInfo()

    plate_number = plate_number.strip().upper()
    if len(plate_number) < 6:
        plate_info._error = "Invalid plate length. Too short."
        return plate_info

    pattern = r"^([ABEKMHOPCTYX])(\d{3})([ABEKMHOPCTYX])([ABEKMHOPCTYX])(\d{2,3})$"
    match = re.match(pattern, plate_number)
    if not match:
        plate_info._error = "Invalid plate format. Expected format like 'H744BH977'."
        return plate_info

    letter1, digits, letter2, letter3, region_code = match.groups()
    for letter in [letter1, letter2, letter3]:
        if letter not in VALID_LETTERS:
            plate_info._error = f"Invalid letter '{letter}'. Only {', '.join(VALID_LETTERS)} are allowed."
            return plate_info

    if digits == '000':
        plate_info._error = "Digits '000' are not allowed."
        return plate_info

    region_name = None
    for name, codes in REGION_CODES.items():
        if region_code in codes:
            region_name = name
            break

    if not region_name:
        plate_info._error = f"Unknown region code: {region_code}"
        return plate_info

    plate_info._valid = True
    plate_info._letter1 = letter1
    plate_info._digits = digits
    plate_info._letter2 = letter2
    plate_info._letter3 = letter3
    plate_info._region_code = region_code
    plate_info._region_name = region_name

    digit_value = int(digits)
    letters = letter1 + letter2 + letter3
    for (gov_letters, (digits_from, digits_to), gov_region), (description, forbidden, advantage, significance) in GOVERNMENT_CODES.items():
        if (letters == gov_letters and 
            digits_from <= digit_value <= digits_to and 
            region_code == gov_region):
            plate_info._government_info = {
                'description': description,
                'forbidden_to_buy': bool(forbidden),
                'road_advantage': bool(advantage),
                'significance_level': significance
            }
            break

    return plate_info

def is_valid_russian_plate(plate_number):
    return parse_license_plate(plate_number).valid

# Example usage
if __name__ == "__main__":
    test_plates = [
        "H744BH977", "T333HX777", "M081TX797", "P700TT790",
        "A000BC78", "Z123BC78", "A12BC78", "ABC12399"
    ]

    for plate in test_plates:
        info = parse_license_plate(plate)
        if info.valid:
            print(f"✅ {plate}: Valid plate from {info.region_name}")
            if info.government_info:
                gov_info = info.government_info
                print(f"   🏛️ {gov_info['description']}")
                print(f"   - Forbidden to buy: {'Yes' if gov_info['forbidden_to_buy'] else 'No'}")
                print(f"   - Road advantage: {'Yes' if gov_info['road_advantage'] else 'No'}")
                print(f"   - Significance level: {gov_info['significance_level']}/10")
        print()
