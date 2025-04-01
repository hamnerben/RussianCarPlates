from supplemental_english import REGION_CODES, GOVERNMENT_CODES

def parse_license_plate(plate_number):
    """
    Parse a Russian license plate and return information about it.
    
    The correct format is: LNNNNLL or LNNNNLRR or LNNNNLRRR where:
    L = letter from the set 'ABEKMHOPCTYX'
    N = digit (cannot be '000' for the numeric part)
    R = region code digits
    
    Examples of valid plates: H744BH977, T333HX777, M081TX797, P700TT790
    
    Args:
        plate_number (str): The license plate to parse
    
    Returns:
        dict: Dictionary containing information about the license plate:
              - valid (bool): Whether the plate follows the correct format
              - letter1 (str): First letter
              - digits (str): The digits (usually 3)
              - letter2 (str): Second letter
              - letter3 (str): Third letter
              - region_code (str): The region code
              - region_name (str): The name of the region associated with the code
              - government_info (dict or None): Information about government codes if applicable
              - error (str or None): Error message if the plate is invalid
    """
    # Define valid Russian license plate letters (Cyrillic letters that look like Latin)
    VALID_LETTERS = set('ABEKMHOPCTYX')
    
    result = {
        'valid': False,
        'letter1': None,
        'digits': None,
        'letter2': None, 
        'letter3': None,
        'region_code': None,
        'region_name': None,
        'government_info': None,
        'error': None
    }
    
    # Remove spaces and standardize format
    plate_number = plate_number.strip().upper()
    
    # Basic format check
    if len(plate_number) < 6:
        result['error'] = "Invalid plate length. Too short."
        return result
    
    # Extract parts using regex pattern
    import re
    pattern = r"^([ABEKMHOPCTYX])(\d{3})([ABEKMHOPCTYX])([ABEKMHOPCTYX])(\d{2,3})$"
    match = re.match(pattern, plate_number)
    
    if not match:
        result['error'] = "Invalid plate format. Expected format like 'H744BH977'."
        return result
    
    letter1, digits, letter2, letter3, region_code = match.groups()
    
    # Validate letters
    for letter in [letter1, letter2, letter3]:
        if letter not in VALID_LETTERS:
            result['error'] = f"Invalid letter '{letter}'. Only {', '.join(VALID_LETTERS)} are allowed."
            return result
    
    # Check '000' is not allowed
    if digits == '000':
        result['error'] = "Digits '000' are not allowed."
        return result
    
    # Check if region code exists
    region_name = None
    for name, codes in REGION_CODES.items():
        if region_code in codes:
            region_name = name
            break
    
    if not region_name:
        result['error'] = f"Unknown region code: {region_code}"
        return result
    
    # At this point, the plate is valid
    result['valid'] = True
    result['letter1'] = letter1
    result['digits'] = digits
    result['letter2'] = letter2
    result['letter3'] = letter3
    result['region_code'] = region_code
    result['region_name'] = region_name
    
    # Check for government codes
    digit_value = int(digits)
    letters = letter1 + letter2 + letter3
    
    # Reconstruct the three-letter combination for government code lookup
    for (gov_letters, (digits_from, digits_to), gov_region), (description, forbidden, advantage, significance) in GOVERNMENT_CODES.items():
        if (letters == gov_letters and 
            digits_from <= digit_value <= digits_to and 
            region_code == gov_region):
            result['government_info'] = {
                'description': description,
                'forbidden_to_buy': bool(forbidden),
                'road_advantage': bool(advantage),
                'significance_level': significance
            }
            break
    
    return result


def is_valid_russian_plate(plate_number):
    """
    Simple function to check if a plate number is valid according to Russian format.
    
    Args:
        plate_number (str): The license plate to check
        
    Returns:
        bool: True if the plate is valid, False otherwise
    """
    return parse_license_plate(plate_number)['valid']


# Example usage
if __name__ == "__main__":
    # Test with the provided example plates
    test_plates = [
        "H744BH977",  # Valid
        "T333HX777",  # Valid
        "M081TX797",  # Valid 
        "P700TT790",  # Valid
        "A000BC78",   # Invalid (000)
        "Z123BC78",   # Invalid letter
        "A12BC78",    # Invalid format
        "ABC12399"    # Invalid format
    ]
    
    for plate in test_plates:
        result = parse_license_plate(plate)
        
        if result['valid']:
            print(f"✅ {plate}: Valid plate from {result['region_name']}")
            if result['government_info']:
                gov_info = result['government_info']
                print(f"   🏛️ {gov_info['description']}")
                print(f"   - Forbidden to buy: {'Yes' if gov_info['forbidden_to_buy'] else 'No'}")
                print(f"   - Road advantage: {'Yes' if gov_info['road_advantage'] else 'No'}")
                print(f"   - Significance level: {gov_info['significance_level']}/10")
        else:
            print(f"❌ {plate}: Invalid plate - {result['error']}")
        print()