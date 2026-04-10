"""
Local AI Knowledge Base for Crop Recommendations
Provides intelligent farming advice without external APIs
"""

CROP_DATABASE = {
    "rice": {
        "optimal_conditions": {
            "nitrogen": (40, 120),
            "phosphorus": (10, 80),
            "potassium": (10, 100),
            "temperature": (20, 30),
            "humidity": (50, 90),
            "ph": (5.5, 6.5),
            "rainfall": (100, 300)
        },
        "explanation": "Rice thrives in these conditions with warm temperatures and consistent moisture. The moderate nutrient levels and neutral pH are ideal for optimal growth.",
        "tips": [
            "Maintain consistent water levels (5-10 cm) throughout growing season",
            "Apply nitrogen fertilizer in 2-3 splits for better utilization",
            "Use certified seeds to prevent disease transmission",
            "Practice proper crop rotation to maintain soil health",
            "Monitor for pests like stem borers and leaf folders"
        ],
        "benefits": [
            "High yield potential (5-8 tons/hectare)",
            "Excellent market demand and commercial viability",
            "Good soil improvement with organic matter decomposition",
            "Can be grown in various soil types with proper management",
            "Nutritious staple grain with multiple uses"
        ]
    },
    
    "papaya": {
        "optimal_conditions": {
            "nitrogen": (30, 100),
            "phosphorus": (15, 60),
            "potassium": (20, 120),
            "temperature": (21, 32),
            "humidity": (60, 80),
            "ph": (5.5, 7.0),
            "rainfall": (50, 200)
        },
        "explanation": "Papaya prefers warm tropical conditions with moderate rainfall. The balanced nutrients and slightly acidic pH promote rapid vegetative growth and fruit production.",
        "tips": [
            "Provide strong support structures as trees mature and bear fruit",
            "Apply organic mulch to retain soil moisture and regulate temperature",
            "Use drip irrigation for efficient water management",
            "Prune regularly to maintain tree shape and improve air circulation",
            "Harvest fruits when they show yellow coloration (3/4 colored)"
        ],
        "benefits": [
            "Fast-growing with fruit production in 9-11 months",
            "High nutritional value (Vitamin C, enzymes, minerals)",
            "Excellent returns in tropical and subtropical regions",
            "Year-round production in suitable climate",
            "Uses for fresh fruit, juice, papain enzyme extraction"
        ]
    },
    
    "maize": {
        "optimal_conditions": {
            "nitrogen": (80, 140),
            "phosphorus": (20, 80),
            "potassium": (15, 90),
            "temperature": (21, 28),
            "humidity": (40, 70),
            "ph": (6.0, 7.5),
            "rainfall": (80, 200)
        },
        "explanation": "Maize requires higher nitrogen for robust growth and requires warm temperatures. The slightly higher potassium helps in grain filling and disease resistance.",
        "tips": [
            "Plant at correct population (50,000-80,000 plants/hectare)",
            "Use quality hybrid seeds with good germination rate",
            "Apply 50% nitrogen before planting, 50% at V4-V6 stage",
            "Control weeds effectively during critical growth period (3-8 weeks)",
            "Provide adequate irrigation, especially during grain filling"
        ],
        "benefits": [
            "High yielding crop (6-10 tons/hectare)",
            "Multiple uses: grain, fodder, industrial applications",
            "Good income stability with strong market demand",
            "Fits well in crop rotation systems",
            "Relatively short duration (120-150 days)"
        ]
    },
    
    "tomato": {
        "optimal_conditions": {
            "nitrogen": (50, 150),
            "phosphorus": (30, 100),
            "potassium": (40, 150),
            "temperature": (20, 27),
            "humidity": (50, 75),
            "ph": (6.0, 6.8),
            "rainfall": (50, 150)
        },
        "explanation": "Tomato needs balanced nutrition with emphasis on potassium for fruit quality. Moderate warmth and humidity prevent fungal diseases while enabling productive flowering.",
        "tips": [
            "Use disease-resistant varieties suited to your region",
            "Install drip irrigation to minimize fungal diseases",
            "Apply balanced fertilizer every 2-3 weeks during growth",
            "Prune lower leaves to improve air circulation and reduce disease",
            "Harvest when fruits show full red color but are still firm"
        ],
        "benefits": [
            "High market value and year-round demand",
            "Quick returns (60-70 days to first harvest)",
            "Suitable for both field and protected cultivation",
            "Rich in lycopene, vitamins, and antioxidants",
            "Multiple harvests from single plant"
        ]
    },
    
    "wheat": {
        "optimal_conditions": {
            "nitrogen": (60, 120),
            "phosphorus": (20, 60),
            "potassium": (20, 80),
            "temperature": (15, 25),
            "humidity": (30, 60),
            "ph": (6.5, 7.5),
            "rainfall": (80, 150)
        },
        "explanation": "Wheat thrives in cooler conditions with moderate nutrients. The slightly alkaline pH is ideal, and lower humidity reduces disease pressure.",
        "tips": [
            "Select winter or spring varieties based on your climate",
            "Prepare seedbed properly to ensure good germination",
            "Apply nitrogen in 2-3 splits: at tillering and grain filling",
            "Control weeds during early growth stages",
            "Harvest when grain moisture drops to 12-13%"
        ],
        "benefits": [
            "Stable staple crop with consistent global demand",
            "Good storage potential and shelf life",
            "Multiple uses: bread, pasta, livestock feed",
            "Important for food security",
            "Relatively low input compared to other crops"
        ]
    },
    
    "cotton": {
        "optimal_conditions": {
            "nitrogen": (80, 130),
            "phosphorus": (20, 60),
            "potassium": (30, 120),
            "temperature": (25, 35),
            "humidity": (50, 75),
            "ph": (6.0, 7.0),
            "rainfall": (50, 150)
        },
        "explanation": "Cotton requires warm conditions, adequate but not excessive water, and balanced nutrients. Higher potassium improves fiber quality and strength.",
        "tips": [
            "Use quality cottonseed from certified sources",
            "Maintain plant population of 40,000-60,000 plants/hectare",
            "Apply split doses of nitrogen for efficient uptake",
            "Implement integrated pest management strategies",
            "Harvest bolls at proper maturity before rainfall"
        ],
        "benefits": [
            "High-value cash crop with strong export market",
            "Long growing season allows sequential planting",
            "Cotton byproducts (oil, meal) have commercial value",
            "Fiber quality commands premium prices",
            "Good income potential in suitable climates"
        ]
    },
    
    "banana": {
        "optimal_conditions": {
            "nitrogen": (50, 120),
            "phosphorus": (20, 60),
            "potassium": (100, 200),
            "temperature": (24, 32),
            "humidity": (75, 90),
            "ph": (5.5, 7.0),
            "rainfall": (150, 300)
        },
        "explanation": "Banana needs high potassium for fruit development and quality. The warm, humid environment prevents stress, while good rainfall supports continuous growth.",
        "tips": [
            "Select disease-resistant varieties (Cavendish, Nendran, etc.)",
            "Use healthy suckers or tissue culture plants",
            "Apply high potassium fertilizer for better fruit quality",
            "Provide mulch to conserve moisture and add organic matter",
            "Harvest bunches when fingers transition from angular to rounded"
        ],
        "benefits": [
            "Perennial crop providing continuous income",
            "Year-round harvest potential in suitable climate",
            "High nutritional value and global market demand",
            "Quick revenue generation (8-10 months to fruit)",
            "Uses: fresh fruit, processing, animal feed"
        ]
    },

    "mango": {
        "optimal_conditions": {
            "nitrogen": (40, 100),
            "phosphorus": (15, 50),
            "potassium": (30, 100),
            "temperature": (24, 32),
            "humidity": (50, 70),
            "ph": (5.5, 7.5),
            "rainfall": (80, 200)
        },
        "explanation": "Mango prefers warm tropical climate with a dry season for flowering. Moderate nutrients and well-drained soil prevent waterlogging and disease.",
        "tips": [
            "Plant grafted trees for uniform fruiting and superior quality",
            "Apply organic mulch to conserve moisture",
            "Prune after harvest to maintain tree shape and vigor",
            "Apply calcium and boron micro-nutrients for quality",
            "Practice proper thinning to get larger, premium fruits"
        ],
        "benefits": [
            "Long-lived productive tree (25+ years)",
            "Premium fruit commanding high market prices",
            "Low maintenance after establishment",
            "Rich in vitamins A, C, and fiber",
            "Processing potential: juices, pulp, leather"
        ]
    },

    "coconut": {
        "optimal_conditions": {
            "nitrogen": (50, 100),
            "phosphorus": (15, 40),
            "potassium": (50, 150),
            "temperature": (24, 32),
            "humidity": (70, 90),
            "ph": (5.5, 8.0),
            "rainfall": (150, 300)
        },
        "explanation": "Coconut thrives in tropical coastal areas with high rainfall and humidity. High potassium is critical for nut development and kernel quality.",
        "tips": [
            "Select tall or dwarf varieties based on available space",
            "Apply potassium-rich fertilizers regularly for quality nuts",
            "Maintain good drainage to prevent root rot",
            "Provide supplementary irrigation in dry periods",
            "Harvest mature nuts at 11-12 month stages"
        ],
        "benefits": [
            "Multiple products: coconut, copra, oil, coirandcoir",
            "Long productive lifespan (60+ years)",
            "Excellent returns with processing value-addition",
            "Uses in cosmetics, food, industrial applications",
            "Climate-resilient in coastal regions"
        ]
    }
}

def get_crop_advice(crop_name, nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """
    Generate personalized farming advice based on crop and conditions
    """
    crop_name = crop_name.lower()
    
    if crop_name not in CROP_DATABASE:
        return {
            "explanation": f"Information for {crop_name.capitalize()} is not available in our knowledge base.",
            "tips": ["Consult local agricultural extension services for optimal practices"],
            "benefits": ["Conduct soil tests to determine crop suitability"]
        }
    
    crop_info = CROP_DATABASE[crop_name]
    optimal = crop_info["optimal_conditions"]
    
    # Generate condition-based explanation
    explanation = crop_info["explanation"]
    
    # Check how conditions match optimal ranges
    condition_matches = []
    if optimal["nitrogen"][0] <= nitrogen <= optimal["nitrogen"][1]:
        condition_matches.append("nitrogen levels are optimal")
    if optimal["phosphorus"][0] <= phosphorus <= optimal["phosphorus"][1]:
        condition_matches.append("phosphorus content is suitable")
    if optimal["potassium"][0] <= potassium <= optimal["potassium"][1]:
        condition_matches.append("potassium levels are ideal")
    if optimal["temperature"][0] <= temperature <= optimal["temperature"][1]:
        condition_matches.append("temperature conditions are favorable")
    if optimal["humidity"][0] <= humidity <= optimal["humidity"][1]:
        condition_matches.append("humidity levels support growth")
    if optimal["ph"][0] <= ph <= optimal["ph"][1]:
        condition_matches.append("soil pH is appropriate")
    if optimal["rainfall"][0] <= rainfall <= optimal["rainfall"][1]:
        condition_matches.append("rainfall pattern is conducive")
    
    # Enhance explanation with matching conditions
    if condition_matches:
        explanation += f" Your farm conditions show that {', '.join(condition_matches)}, which provides excellent growing potential for {crop_name.capitalize()}."
    
    return {
        "explanation": explanation,
        "tips": crop_info["tips"],
        "benefits": crop_info["benefits"]
    }
