def fish_details(fish_class):
    fish_dict = {
        "Black Sea Sprat": "The Black Sea sprat, Clupeonella cultriventris, is a small fish of the herring family, Clupeidae. It is found in the Black Sea and Sea of Azov and rivers of its basins: Danube, Dnister, Dnipro (Ukraine), Southern Bug, Don, Kuban. It has white-grey flesh and silver-grey scales. A typical size is 10 cm (maximum 15 cm) The life span is of up to 5 years. The peak of its spawning is in April and it can be found in enormous shoals in sea-shores, filled all-round coastal shallows, moving quickly back in the sea at a depth of 6–30 metres. Used for food; it has around 12% fat in flesh.",
        "Gilt-Head Bream": "The gilt-head bream, known as Orata in antiquity and still today in Italy, is a fish of the bream family Sparidae found in the Mediterranean Sea and the eastern coastal regions of the North Atlantic Ocean. It commonly reaches about 35 centimetres in length, but may reach 70 cm and weigh up to about 7.36 kilograms.",
        "Horse Mackerel": "The Atlantic horse mackerel, also known as the European horse mackerel or common scad, is a species of jack mackerel in the family Carangidae, the jacks, pompanos and trevallies. It is found in the eastern Atlantic Ocean off Europe and Africa and into the south-eastern Indian Ocean.",
        "Red Mullet": 'The red mullets or surmullets are two species of goatfish, Mullus barbatus and Mullus surmuletus, found in the Mediterranean Sea, east North Atlantic Ocean, and the Black Sea. Both "red mullet" and "surmullet" can also refer to the Mullidae in general.',
        "Red Sea Bream": "Pagrus major or red seabream is a fish species in the family Sparidae. It is also known by its Japanese name, madai. The fish has high culinary and cultural importance in Japan, and is also frequently eaten in Korea and Taiwan.",
        "Sea Bass": "Sea bass is a common name for a variety of different species of marine fish. Many fish species of various families have been called sea bass.",
        "Shrimp": "Shrimp are decapod crustaceans with elongated bodies and a primarily swimming mode of locomotion – most commonly Caridea and Dendrobranchiata. More narrow definitions may be restricted to Caridea, to smaller species of either group or to only the marine species.",
        "Striped Red Mullet": "The striped red mullet or surmullet is a species of goatfish found in the Mediterranean Sea, eastern North Atlantic Ocean, and the Black Sea. They can be found in water as shallow as 5 metres or as deep as 409 metres depending upon the portion of their range that they are in.",
        "Trout": "Trout are species of freshwater fish belonging to the genera Oncorhynchus, Salmo and Salvelinus, all of the subfamily Salmoninae of the family Salmonidae. The word trout is also used as part of the name of some non-salmonid fish such as Cynoscion nebulosus, the spotted seatrout or speckled trout."
    }

    return fish_dict[fish_class]


def plant_details(plant_class):
    desc = ""
    if "Apple" in plant_class:
        desc = "Apple trees are cultivated worldwide and are the most widely grown species in the genus Malus. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found today."
    elif "Blueberry" in plant_class:
        desc = "Blueberries are a widely distributed and widespread group of perennial flowering plants with blue or purple berries. They are classified in the section Cyanococcus within the genus Vaccinium. Vaccinium also includes cranberries, bilberries, huckleberries and Madeira blueberries."
    elif "Cherry" in plant_class:
        desc = "A cherry is the fruit of many plants of the genus Prunus, and is a fleshy drupe. Commercial cherries are obtained from cultivars of several species, such as the sweet Prunus avium and the sour Prunus cerasus."
    elif "Corn" in plant_class:
        desc = "Maize, also known as corn, is a cereal grain first domesticated by indigenous peoples in southern Mexico about 10,000 years ago. The leafy stalk of the plant produces pollen inflorescences and separate ovuliferous inflorescences called ears that when fertilized yield kernels or seeds, which are fruits."
    elif "Grape" in plant_class:
        desc = "A grape is a fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis. Grapes can be eaten fresh as table grapes, used for making wine, jam, grape juice, jelly, grape seed extract, vinegar, and grape seed oil, or dried as raisins, currants and sultanas. "
    elif "Orange" in plant_class:
        desc = "An orange is a fruit of various citrus species in the family Rutaceae; it primarily refers to Citrus × sinensis, which is also called sweet orange, to distinguish it from the related Citrus × aurantium, referred to as bitter orange."
    elif "Peach" in plant_class:
        desc = "The peach is a deciduous tree first domesticated and cultivated in Zhejiang province of Eastern China. It bears edible juicy fruits with various characteristics, most called peaches and others, nectarines."
    elif "Pepper" in plant_class:
        desc = 'The bell pepper is the fruit of plants in the Grossum cultivar group of the species Capsicum annuum. Cultivars of the plant produce fruits in different colors, including red, yellow, orange, green, white, and purple. Bell peppers are sometimes grouped with less pungent chili varieties as "sweet peppers".'
    elif "Potato" in plant_class:
        desc = "The potato is a starchy tuber of the plant Solanum tuberosum and is a root vegetable native to the Americas. The plant is a perennial in the nightshade family Solanaceae. Wild potato species can be found from the southern United States to southern Chile."
    elif "Raspberry" in plant_class:
        desc = "The raspberry is the edible fruit of a multitude of plant species in the genus Rubus of the rose family, most of which are in the subgenus Idaeobatus. The name also applies to these plants themselves. Raspberries are perennial with woody stems."
    elif "Soybean" in plant_class:
        desc = "The soybean, soy bean, or soya bean is a species of legume native to East Asia, widely grown for its edible bean, which has numerous uses. Traditional unfermented food uses of soybeans include soy milk, from which tofu and tofu skin are made."
    elif "Squash" in plant_class:
        desc = "Cucurbita is a genus of herbaceous vegetables in the gourd family, Cucurbitaceae native to the Andes and Mesoamerica. Five species are grown worldwide for their edible vegetable, variously known as squash, pumpkin, or gourd, depending on species, variety, and local parlance, and for their seeds."
    elif "Strawberry" in plant_class:
        desc = "The garden strawberry is a widely grown hybrid species of the genus Fragaria, collectively known as the strawberries, which are cultivated worldwide for their fruit. The fruit is widely appreciated for its characteristic aroma, bright red color, juicy texture, and sweetness."
    elif "Tomato" in plant_class:
        desc = "The tomato is the edible berry of the plant Solanum lycopersicum, commonly known as the tomato plant. The species originated in western South America and Central America. The Mexican Nahuatl word tomatl gave rise to the Spanish word tomate, from which the English word tomato derived."
    else:
        desc = "Plant not found"

    return desc
