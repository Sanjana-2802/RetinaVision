"""
Personalized Health, Nutrition & Eye Care Recommendation Engine.
Formulates tailored dietary blueprints, vitamin regimens, foods to avoid,
ergonomic screen habits, and lifestyle interventions based on identified eye disease and severity.
"""

DISEASE_RECOMMENDATIONS = {
    "Diabetic Retinopathy": {
        "nutrients": [
            {"name": "Lutein & Zeaxanthin", "benefit": "Filters high-energy blue wavelengths and strengthens macular pigment optical density (MPOD).", "sources": "Kale, spinach, collard greens, egg yolks"},
            {"name": "Alpha-Lipoic Acid (ALA)", "benefit": "Potent antioxidant that improves retinal insulin sensitivity and suppresses capillary leakage.", "sources": "Spinach, broccoli, Brussels sprouts, organ meats"},
            {"name": "Omega-3 Fatty Acids (EPA/DHA)", "benefit": "Decreases pathological retinal neovascularization and vascular endothelial growth factor (VEGF).", "sources": "Wild salmon, sardines, chia seeds, flaxseed oil"},
            {"name": "Vitamin C & Bioflavonoids", "benefit": "Reinforces microvascular capillary walls and inhibits basement membrane thickening.", "sources": "Guava, amla (Indian gooseberry), bell peppers, citrus fruits"}
        ],
        "meal_plan": {
            "breakfast": "Steel-cut oatmeal topped with ground flaxseed, walnuts, and fresh blueberries with unsweetened almond milk.",
            "lunch": "Steamed rainbow salad with baby spinach, grilled wild salmon or tofu, avocado, pumpkin seeds, and extra virgin olive oil.",
            "snack": "A handful of raw walnuts, roasted almonds, and cucumber slices with hummus.",
            "dinner": "Steamed broccoli, grilled chicken breast or tempeh, and a small portion of quinoa with turmeric and black pepper."
        },
        "foods_to_avoid": [
            {"item": "Refined Sugars & Syrups", "reason": "Causes acute postprandial glycemic spikes that injure delicate retinal pericytes."},
            {"item": "Ultra-Processed Trans Fats", "reason": "Exacerbates systemic vascular endothelial inflammation and microaneurysm leakage."},
            {"item": "High Sodium Processed Foods", "reason": "Elevates systemic blood pressure, exacerbating macular edema and exudate deposition."}
        ],
        "eye_care_routine": [
            "Strict adherence to the 20-20-20 rule: Every 20 minutes, gaze at an object 20 feet away for 20 seconds.",
            "Avoid vigorous straining or inverted yoga postures if severe proliferative changes are observed.",
            "Maintain artificial tear lubricating drops (preservative-free) for secondary dry eye syndrome."
        ],
        "lifestyle": [
            "Maintain strict glycemic control: Target HbA1c < 7.0% (as advised by your endocrinologist).",
            "Monitor blood pressure daily: Target < 130/80 mmHg to reduce shearing stress on retinal capillaries.",
            "Comprehensive dilated fundus exam every 3 to 6 months depending on severity progression."
        ]
    },
    "Glaucoma": {
        "nutrients": [
            {"name": "Nitrate-Rich Leafy Greens", "benefit": "Enhances nitric oxide bio-availability, improving ocular blood flow and trabecular meshwork drainage.", "sources": "Kale, arugula, beet greens, spinach"},
            {"name": "Ginkgo Biloba & Flavonoids", "benefit": "Neuroprotective effect on retinal ganglion cells and improves retrobulbar hemodynamics.", "sources": "Dark chocolate (85%+), green tea, blueberries"},
            {"name": "Magnesium", "benefit": "Relaxes retinal arteriolar vasospasm and optimizes ocular perfusion pressure.", "sources": "Pumpkin seeds, almonds, black beans, dark leafy greens"},
            {"name": "Omega-3 Fatty Acids", "benefit": "Supports optimal neuroretinal nerve fiber layer hydration and anti-inflammatory tone.", "sources": "Walnuts, mackerel, hemp seeds"}
        ],
        "meal_plan": {
            "breakfast": "Green power smoothie with raw spinach, celery, green apple, chia seeds, and coconut water.",
            "lunch": "Quinoa bowl with roasted beetroot, dark leafy greens, edamame, and tahini-lemon dressing.",
            "snack": "Small square of 85% dark cacao chocolate and a cup of organic sencha green tea.",
            "dinner": "Baked trout or lentil curry with steamed asparagus and brown basmati rice."
        },
        "foods_to_avoid": [
            {"item": "Excessive Caffeine (>200mg/day)", "reason": "Can cause transient spikes in Intraocular Pressure (IOP)."},
            {"item": "Rapid High-Volume Fluid Ingestion", "reason": "Gulping more than 500ml of fluid in a short window spikes IOP."},
            {"item": "Excess Table Salt", "reason": "Disturbs fluid osmolarity and microcirculation in the optic nerve head."}
        ],
        "eye_care_routine": [
            "Strict compliance with prescribed hypotensive eye drops at exact daily hours without missing doses.",
            "Avoid neckties tied too tightly and tight swimming goggles which raise episcleral venous pressure.",
            "Perform gentle breathing exercises during physical exertion; avoid holding breath (Valsalva maneuver)."
        ],
        "lifestyle": [
            "Regular aerobic exercise (brisk walking 30 mins/day) naturally lowers intraocular pressure.",
            "Elevate head of bed by 20 to 30 degrees while sleeping to prevent nocturnal IOP spikes.",
            "Regular Humphrey visual field perimetry and OCT optic nerve fiber scans every 6 months."
        ]
    },
    "Cataract": {
        "nutrients": [
            {"name": "Glutathione Boosters", "benefit": "Key endogenous peptide preventing crystalline lens protein cross-linking and opacification.", "sources": "Garlic, onions, asparagus, whey protein"},
            {"name": "Vitamin C (Ascorbic Acid)", "benefit": "Maintains aqueous humor antioxidant capacity protecting the lens from UV photo-oxidation.", "sources": "Bell peppers, kiwis, strawberries, oranges, amla"},
            {"name": "Vitamin E (Alpha-Tocopherol)", "benefit": "Lipid-soluble antioxidant shielding lens cell membranes from lipid peroxidation.", "sources": "Sunflower seeds, almonds, avocados"},
            {"name": "Carotenoids (Beta-Carotene)", "benefit": "Scavenges singlet oxygen free radicals generated by ambient UV rays.", "sources": "Carrots, sweet potatoes, butternut squash"}
        ],
        "meal_plan": {
            "breakfast": "Chia seed pudding with fresh kiwi, crushed almonds, and orange zest.",
            "lunch": "Mediterranean chickpea salad with diced bell peppers, red onions, garlic vinaigrette, and fresh parsley.",
            "snack": "Handful of sunflower seeds and raw carrot sticks with guacamole.",
            "dinner": "Grilled salmon or roasted tofu with sautéed asparagus, garlic, and sweet potato mash."
        },
        "foods_to_avoid": [
            {"item": "Deep-Fried & Charred Foods", "reason": "High levels of Advanced Glycation End-products (AGEs) accelerate lens clouding."},
            {"item": "High-Fructose Corn Syrup", "reason": "Promotes sorbitol accumulation in crystalline lens fibers causing osmotic swelling."},
            {"item": "Excess Alcohol", "reason": "Depletes hepatic glutathione stores critical for ocular lens defense."}
        ],
        "eye_care_routine": [
            "Wear 100% UV-blocking (UV400) polarized sunglasses whenever outdoors during daylight hours.",
            "Ensure adequate ambient lighting while reading to prevent eye strain under cloudy lens media.",
            "Limit night driving if halos, glare, and contrast attenuation compromise visual safety."
        ],
        "lifestyle": [
            "Cessation of cigarette smoking (smoking increases cataract formation risk twofold).",
            "Periodic refraction updates to optimize distance and reading spectacle power.",
            "Consult an anterior segment surgeon if functional visual acuity falls below daily lifestyle requirements."
        ]
    },
    "Age-Related Macular Degeneration (AMD)": {
        "nutrients": [
            {"name": "AREDS-2 Formulation Elements", "benefit": "Clinically proven to reduce progression risk from intermediate to advanced neovascular AMD.", "sources": "Lutein (10mg), Zeaxanthin (2mg), Vit C, Vit E, Zinc, Copper"},
            {"name": "Zinc & Copper Balance", "benefit": "Highly concentrated in choroid and retina, essential for retinal pigment epithelium metalloenzymes.", "sources": "Pumpkin seeds, oysters, lentils, cashews"},
            {"name": "Anthocyanins", "benefit": "Strengthens rod/cone photoreceptor regeneration and promotes dark adaptation.", "sources": "Wild bilberries, blackberries, blackcurrants"},
            {"name": "Omega-3 (DHA Heavy)", "benefit": "DHA constitutes 50-60% of photoreceptor outer segment membrane fatty acids.", "sources": "Fatty cold-water fish, algae-based DHA supplements"}
        ],
        "meal_plan": {
            "breakfast": "Poached eggs on toasted whole grain sourdough with sautéed baby spinach and avocado slices.",
            "lunch": "Warm lentil stew with kale, carrots, turmeric, and a sprinkle of toasted pumpkin seeds.",
            "snack": "A cup of fresh blackberries and a small handful of raw Brazil nuts and cashews.",
            "dinner": "Pan-seared cod or tempeh over a bed of steamed Romanesco broccoli and wild rice."
        },
        "foods_to_avoid": [
            {"item": "High Glycemic Index Starches", "reason": "White bread, pastries, and processed potatoes correlate with faster AMD progression."},
            {"item": "Hydrogenated Vegetable Oils", "reason": "Displaces essential retinal DHA in photoreceptor lipid bilayers."},
            {"item": "Processed Meats (Bacon, Sausages)", "reason": "High nitrates and saturated fats compromise choroidal choriocapillaris circulation."}
        ],
        "eye_care_routine": [
            "Weekly self-monitoring with an Amsler Grid (monocularly) to catch waviness or dark spots early.",
            "Wear amber-tinted lenses to minimize blue-light induced photo-oxidative stress on RPE cells.",
            "Optimize reading lights with high color rendering index (CRI > 90) task lighting."
        ],
        "lifestyle": [
            "Immediate smoking cessation is imperative (smoking is the strongest modifiable risk factor for AMD).",
            "Monitor choroidal perfusion through regular cardiovascular workouts.",
            "Schedule regular Optical Coherence Tomography (OCT) scans to rule out choroidal neovascularization."
        ]
    },
    "Hypertensive Retinopathy": {
        "nutrients": [
            {"name": "Potassium-Rich Foods", "benefit": "Counteracts vascular sodium retention and induces vascular smooth muscle relaxation.", "sources": "Avocados, bananas, spinach, coconut water, white beans"},
            {"name": "Nitric Oxide Promoters (Citrulline/Nitrates)", "benefit": "Dilates constricted retinal arterioles and lowers peripheral vascular resistance.", "sources": "Watermelon, beetroot, arugula, garlic"},
            {"name": "Magnesium Glycinate", "benefit": "Prevents retinal precapillary vasospasm and regulates intracellular calcium influx.", "sources": "Swiss chard, almonds, cacao nibs"},
            {"name": "Polyphenols & Resveratrol", "benefit": "Protects retinal vascular endothelium from high shear-stress oxidative damage.", "sources": "Red grapes, blueberries, dark cherries"}
        ],
        "meal_plan": {
            "breakfast": "Avocado and sliced tomato on seeded multigrain bread with a cup of hibiscus herbal tea.",
            "lunch": "DASH diet roasted beet and goat cheese salad with walnuts and balsamic olive oil dressing.",
            "snack": "Fresh watermelon slices or celery sticks with unsalted almond butter.",
            "dinner": "Herb-crusted baked chicken breast or grilled portobello mushroom with steamed green beans and brown rice."
        },
        "foods_to_avoid": [
            {"item": "Excess Dietary Sodium (>1500mg/day)", "reason": "Directly triggers fluid retention and spikes arteriolar pressure onto retinal beds."},
            {"item": "Cured & Smoked Meats", "reason": "Contains heavy sodium and preservatives that stiffen arteriolar walls."},
            {"item": "Energy Drinks & High Caffeine", "reason": "Triggers sudden spikes in peripheral vascular resistance and vasoconstriction."}
        ],
        "eye_care_routine": [
            "Rest eyes periodically in comfortable low-glare lighting to prevent visual fatigue.",
            "Avoid strenuous breath-holding activities that cause sudden blood pressure spikes in ocular vessels.",
            "Use preservative-free ocular lubricants if experiencing hypertensive dry eye."
        ],
        "lifestyle": [
            "Maintain strict blood pressure control with regular sphygmomanometer checks (<120/80 mmHg ideal).",
            "Practice daily 15-minute mindfulness meditation or slow diaphragmatic breathing to reduce sympathetic tone.",
            "Coordinate care between your ophthalmologist and cardiologist for systemic risk reduction."
        ]
    },
    "Normal": {
        "nutrients": [
            {"name": "Balanced Vision Super-Nutrients", "benefit": "Maintains lifelong corneal clarity, retinal health, and optimal visual acuity.", "sources": "Colorful berries, wild salmon, leafy greens, carrots"},
            {"name": "Lutein & Zeaxanthin", "benefit": "Preserves dense macular pigment shield against daily digital screen emissions.", "sources": "Corn, spinach, eggs, bell peppers"},
            {"name": "Omega-3 Fatty Acids", "benefit": "Maintains meibomian gland oil secretion to prevent dry eye and computer fatigue.", "sources": "Chia seeds, flaxseeds, walnuts, fatty fish"},
            {"name": "Hydration Balance", "benefit": "Ensures adequate tear film volume and vitreous body hydration.", "sources": "Pure water, herbal teas, fresh coconut water"}
        ],
        "meal_plan": {
            "breakfast": "Scrambled eggs with sautéed spinach, tomatoes, and whole wheat toast, with fresh orange juice.",
            "lunch": "Mediterranean grain bowl with quinoa, chickpeas, cucumbers, cherry tomatoes, and tahini dressing.",
            "snack": "A fresh organic apple with a handful of mixed walnuts and pumpkin seeds.",
            "dinner": "Pan-seared cod or lentil curry served with steamed broccoli, carrots, and brown rice."
        },
        "foods_to_avoid": [
            {"item": "Excessive Junk & Fast Food", "reason": "Accelerates long-term chronic oxidative stress across microvascular networks."},
            {"item": "Dehydrating Sugary Carbonated Sodas", "reason": "Contributes to systemic dehydration and dry eye symptoms."},
            {"item": "Excessive Late-Night Refined Carbs", "reason": "Disrupts circadian melatonin regulation and ocular surface healing."}
        ],
        "eye_care_routine": [
            "Follow the 20-20-20 rule during screen use: Every 20 mins, look 20 feet away for 20 seconds.",
            "Consciously blink frequently while reading monitors to prevent tear film evaporation.",
            "Position display monitors roughly 20 to 26 inches away, with top of screen slightly below eye level."
        ],
        "lifestyle": [
            "Wear sunglasses with UV400 protection during outdoor daylight activities.",
            "Schedule regular annual preventive comprehensive eye exams.",
            "Ensure 7 to 8 hours of restorative sleep each night for ocular surface rejuvenation."
        ]
    }
}

SEVERITY_CARE_PLANS = {
    "Normal": {
        "urgency": "Routine Preventive Care",
        "badge_color": "emerald",
        "hospital_recommendation": False,
        "follow_up": "Next routine checkup recommended in 12 months.",
        "action_steps": [
            "Continue your healthy diet and balanced visual wellness routine.",
            "Maintain digital eye strain prevention habits (20-20-20 rule).",
            "Wear UV400 sunglasses outdoors."
        ]
    },
    "Mild": {
        "urgency": "Early Stage - Preventative Monitoring",
        "badge_color": "amber",
        "hospital_recommendation": False,
        "follow_up": "Follow-up scan recommended in 3 to 6 months.",
        "action_steps": [
            "Adopt the personalized nutritional blueprint immediately to slow progression.",
            "Control systemic risk factors (blood sugar, blood pressure, lipid profile).",
            "Monitor vision weekly for any subtle changes or blurriness."
        ]
    },
    "Moderate": {
        "urgency": "Intermediate Stage - Clinical Attention Advised",
        "badge_color": "orange",
        "hospital_recommendation": True,
        "follow_up": "Schedule a comprehensive in-person clinical exam within 2 to 4 weeks.",
        "action_steps": [
            "Book an appointment with an ophthalmologist or subspecialist for confirmatory clinical exam.",
            "Begin targeted dietary supplementation under doctor guidance (e.g. AREDS-2 or Omega-3).",
            "Refrain from eye-straining activities and maintain close self-monitoring."
        ]
    },
    "Severe": {
        "urgency": "High Severity - Immediate Specialist Referral",
        "badge_color": "red",
        "hospital_recommendation": True,
        "follow_up": "URGENT: Visit a specialized eye hospital or retina/glaucoma consultant within 24 to 72 hours.",
        "action_steps": [
            "Immediate hospital consultation required to prevent permanent vision compromise.",
            "Avoid strenuous physical exertion or rapid head inversions.",
            "Download your generated clinical report to present directly to the consulting ophthalmologist."
        ]
    },
    "Proliferative / Advanced": {
        "urgency": "Critical Stage - Emergency Ophthalmologic Care",
        "badge_color": "rose",
        "hospital_recommendation": True,
        "follow_up": "EMERGENCY: Seek immediate emergency eye hospital evaluation within 24 hours.",
        "action_steps": [
            "Emergency specialist intervention required (possible anti-VEGF, laser photocoagulation, or surgical consultation).",
            "Do not drive or operate heavy machinery if vision is impaired.",
            "Bring your complete medical history and this AI diagnostic report to the emergency eye clinic."
        ]
    }
}

def get_health_recommendations(disease: str, severity: str) -> dict:
    """Generates tailored nutritional, eye-care, and lifestyle recommendations."""
    disease_key = disease if disease in DISEASE_RECOMMENDATIONS else "Normal"
    severity_key = severity if severity in SEVERITY_CARE_PLANS else "Moderate"

    disease_data = DISEASE_RECOMMENDATIONS[disease_key]
    severity_data = SEVERITY_CARE_PLANS[severity_key]

    return {
        "disease": disease_key,
        "severity": severity_key,
        "nutrients": disease_data["nutrients"],
        "meal_plan": disease_data["meal_plan"],
        "foods_to_avoid": disease_data["foods_to_avoid"],
        "eye_care_routine": disease_data["eye_care_routine"],
        "lifestyle": disease_data["lifestyle"],
        "urgency": severity_data["urgency"],
        "badge_color": severity_data["badge_color"],
        "hospital_recommendation": severity_data["hospital_recommendation"],
        "follow_up": severity_data["follow_up"],
        "action_steps": severity_data["action_steps"]
    }
