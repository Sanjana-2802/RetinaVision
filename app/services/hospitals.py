"""
Nearby Eye Specialists & Tertiary Ophthalmology Hospital Directory.
Provides emergency referral information, specialist sub-fields, ratings,
and emergency hotlines for moderate, severe, and critical eye conditions.
"""

HOSPITALS_DATABASE = [
    {
        "id": "hosp-01",
        "name": "L.V. Prasad Eye Institute (LVPEI) - Center of Excellence",
        "type": "Tertiary Eye Research & Specialty Hospital",
        "city": "Hyderabad, Telangana",
        "address": "Kallam Anji Reddy Campus, Road No. 2, Banjara Hills",
        "distance": "3.8 km away",
        "rating": 4.9,
        "reviews_count": 14200,
        "emergency_24_7": True,
        "phone": "+91 40 6810 2020",
        "emergency_phone": "+91 40 3061 2828",
        "specialties": ["Vitreoretinal Diseases", "Glaucoma", "Cornea & Anterior Segment", "Pediatric Ophthalmology"],
        "recommended_for": ["Diabetic Retinopathy", "Glaucoma", "Age-Related Macular Degeneration (AMD)", "Cataract", "Hypertensive Retinopathy"],
        "map_query": "LV+Prasad+Eye+Institute+Banjara+Hills+Hyderabad"
    },
    {
        "id": "hosp-02",
        "name": "Dr. Agarwal's Eye Hospital - Advanced Retina Center",
        "type": "Super-Specialty Eye Hospital",
        "city": "Hyderabad / Regional Centers",
        "address": "Opp. GVK One Mall, Road No. 1, Banjara Hills",
        "distance": "4.5 km away",
        "rating": 4.8,
        "reviews_count": 8940,
        "emergency_24_7": True,
        "phone": "+91 95949 24026",
        "emergency_phone": "+91 40 4455 6677",
        "specialties": ["Vitreoretina", "Micro-incision Cataract Surgery (MICS)", "Glaucoma Clinic", "Uveitis"],
        "recommended_for": ["Diabetic Retinopathy", "Cataract", "Age-Related Macular Degeneration (AMD)"],
        "map_query": "Dr+Agarwals+Eye+Hospital+Banjara+Hills"
    },
    {
        "id": "hosp-03",
        "name": "Sankara Eye Hospital",
        "type": "Community & Specialty Eye Care Hospital",
        "city": "Hyderabad / Bangalore",
        "address": "Varthur Main Road, Kundalahalli Gate",
        "distance": "7.2 km away",
        "rating": 4.8,
        "reviews_count": 6800,
        "emergency_24_7": True,
        "phone": "+91 80 2854 2727",
        "emergency_phone": "+91 80 2854 2728",
        "specialties": ["Comprehensive Ophthalmology", "Glaucoma Care", "Diabetic Eye Care Clinic", "Cataract"],
        "recommended_for": ["Glaucoma", "Diabetic Retinopathy", "Cataract"],
        "map_query": "Sankara+Eye+Hospital"
    },
    {
        "id": "hosp-04",
        "name": "Aravind Eye Hospital - Retina & Macula Department",
        "type": "Tertiary Referral & Laser Retina Institute",
        "city": "Madurai / Chennai / Coimbatore",
        "address": "1, Anna Nagar, Shenoy Nagar",
        "distance": "12.4 km away",
        "rating": 4.9,
        "reviews_count": 21500,
        "emergency_24_7": True,
        "phone": "+91 452 435 6100",
        "emergency_phone": "+91 452 435 6111",
        "specialties": ["Vitreoretinal Microsurgery", "Anti-VEGF Therapy", "Glaucoma Shunt Surgery", "Ocular Trauma"],
        "recommended_for": ["Diabetic Retinopathy", "Age-Related Macular Degeneration (AMD)", "Glaucoma"],
        "map_query": "Aravind+Eye+Hospital"
    },
    {
        "id": "hosp-05",
        "name": "Maxivision Super Specialty Eye Hospital",
        "type": "Laser & Diagnostic Ophthalmology Center",
        "city": "Hyderabad, Telangana",
        "address": "Somajiguda, Raj Bhavan Road",
        "distance": "5.1 km away",
        "rating": 4.7,
        "reviews_count": 5420,
        "emergency_24_7": False,
        "phone": "+91 40 4940 5000",
        "emergency_phone": "+91 93901 10000",
        "specialties": ["Robotic Laser Cataract", "Retina & Vitreous", "Glaucoma Valve Surgery", "Cornea"],
        "recommended_for": ["Cataract", "Glaucoma", "Hypertensive Retinopathy"],
        "map_query": "Maxivision+Eye+Hospital+Somajiguda"
    }
]

def get_recommended_hospitals(disease: str = None, filter_emergency: bool = False):
    """Filters hospitals based on the detected condition and urgency."""
    results = []
    for hosp in HOSPITALS_DATABASE:
        if filter_emergency and not hosp["emergency_24_7"]:
            continue
        if disease and disease != "Normal":
            if disease in hosp["recommended_for"] or "Comprehensive Ophthalmology" in hosp["specialties"]:
                results.append(hosp)
        else:
            results.append(hosp)
    return results if results else HOSPITALS_DATABASE
