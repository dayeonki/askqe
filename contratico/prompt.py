perturb_synonym_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Il peut s’agir d’un membre du secrétariat ou du personnel clinique, selon le protocole de chaque cabinet.
Perturbed: Il peut s’agir d’un membre du secrétariat ou du personnel médical, selon le protocole de chaque cabinet.

Original: En outre, nous recruterons de nouveaux cabinets de surveillance.
Perturbed: De plus, nous engagerons de nouveaux cabinets de suivi.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_word_order_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the word order. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Il peut s’agir d’un membre du secrétariat ou du personnel clinique, selon le protocole de chaque cabinet.
Perturbed: Il peut s’agir d’un membre du personnel clinique ou du secrétariat, selon le protocole de chaque cabinet.

Original: Nous développerons un observatoire visant à présenter les données à l’échelle nationale ainsi qu’un tableau de bord pour faire des remarques aux cabinets quant à la qualité de leurs données et la collecte d’échantillons virologiques et sérologiques.
Perturbed: Un observatoire sera développé pour présenter les données à l’échelle nationale, accompagné d’un tableau de bord destiné à fournir des remarques aux cabinets concernant la qualité des données et la collecte des échantillons virologiques et sérologiques.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_fr = """Taks: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by making spelling of one or two words wrong. The words should be important words in the sentence but not words like "le", "et", "la" or "des". Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Il peut s’agir d’un membre du secrétariat ou du personnel clinique, selon le protocole de chaque cabinet.
Perturbed: Il peut s’agir d’un membre du serétariat ou du personnel cinique, selon le protocole de chaque cabinet.

Original: En outre, nous recruterons de nouveaux cabinets de surveillance.
Perturbed: En outre, nous recruterons de nouveaux cainets de sureillance.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: si vous pensez que vos symptômes ou problèmes justifient un examen plus approfondi.
Perturbed: si vous pensez que vos symptômes ou problèmes justifient un examen médical plus approfondi.

Original: En cas de réponse affirmative à ces questions de filtrage, le patient doit être prié de ne pas venir au cabinet et de suivre l’organigramme de PHE à la place.
Perturbed: En cas de réponse affirmative à ces questions de filtrage, le patient adulte doit être prié de ne pas venir au cabinet et de suivre l’organigramme de PHE à la place.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_intensifier_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words that changes the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Les symptômes courants comprennent la fièvre, une toux sèche et la fatigue.
Perturbed: Les symptômes courants comprennent la forte fièvre, une toux sèche et la fatigue.

Original: L’essoufflement, le mal de gorge, les maux de tête, les courbatures ou la production d’expectorations comptent parmi les autres symptômes moins courants.
Perturbed: L’essoufflement sévère, le mal de gorge, les maux de tête, les courbatures ou la production d’expectorations comptent parmi les autres symptômes moins courants.

Original: j’ai fait sur le corps autour de la poitrine ?
Perturbed: j’ai fortement fait sur le corps autour de la poitrine ?

Original: Des formulaires de rapport avaient été soumis aux CDC pour 74 439 cas (60,7 %).
Perturbed: Des formulaires de rapport avaient été soumis aux CDC pour un grand total de 74 439 cas (60,7 %).
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding words in the sentence. Please make sure that the added word does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Les symptômes courants comprennent la fièvre, une toux sèche et la fatigue.
Perturbed: Les symptômes courants comprennent la fièvre et des douleurs musculaires, une toux sèche et la fatigue.

Original: L’essoufflement, le mal de gorge, les maux de tête, les courbatures ou la production d’expectorations comptent parmi les autres symptômes moins courants.
Perturbed: L’essoufflement, le mal de gorge, les maux de tête, les courbatures, la production d’expectorations et des troubles digestifs comptent parmi les autres symptômes moins courants.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_omission_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by removing information from the sentence. Remove only one or two words from the sentence. Please make sure that the removed information does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Les symptômes courants comprennent la fièvre, une toux sèche et la fatigue.
Perturbed: Les symptômes courants comprennent la fatigue et la fatigue.

Original: Des recherches sur un vaccin ou un traitement antiviral spécifique sont en cours.
Perturbed: Des recherches sur un traitement antiviral spécifique sont en cours.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_fr = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the affirmative sentence into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Il n'a pas réussi à soulager la douleur avec des médicaments que la compétition interdit aux concurrents de prendre.
Perturbed: Il n'a pas réussi à soulager le plaisir avec des médicaments, que la compétition interdit aux concurrents de prendre.

Original: Le mois dernier, un comité présidentiel a recommandé la démission de l'ancien CEP dans le cadre de mesures visant à pousser le pays vers de nouvelles élections.
Perturbed: Le mois dernier, un comité présidentiel n'a pas recommandé la démission de l'ancien CEP dans le cadre de mesures visant à pousser le pays vers de nouvelles élections.

Original: et votre nez coule-t-il ?
Perturbed: et votre nez danse-t-il ?
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_synonym_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Puede tratarse de un miembro de la secretaría o del personal clínico, según el protocolo de cada consultorio.
Perturbed: Puede tratarse de un miembro de la secretaría o del personal médico, según el protocolo de cada consultorio.

Original: Además, reclutaremos nuevos consultorios de monitoreo.
Perturbed: Asimismo, contrataremos nuevos consultorios de seguimiento.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_word_order_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the word order. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Puede tratarse de un miembro de la secretaría o del personal clínico, según el protocolo de cada consultorio.
Perturbed: Puede tratarse de un miembro del personal clínico o de la secretaría, según el protocolo de cada consultorio.

Original: Desarrollaremos un observatorio para presentar los datos a nivel nacional, así como un tablero de control para hacer observaciones a los consultorios sobre la calidad de sus datos y la recolección de muestras virológicas y serológicas.
Perturbed: Se desarrollará un observatorio para presentar los datos a nivel nacional, junto con un tablero de control destinado a proporcionar observaciones a los consultorios sobre la calidad de los datos y la recolección de muestras virológicas y serológicas.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_es = """Taks: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by making spelling of one or two words wrong. The words should be important words in the sentence but not words like "le", "et", "la" or "des". Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Puede tratarse de un miembro de la secretaría o del personal clínico, según el protocolo de cada consultorio.
Perturbed: Puede tratarse de un miembro de la secretaría o del persnal clínico, según el protcolo de cada consultorio.

Original: Además, reclutaremos nuevos consultorios de monitoreo.
Perturbed: Además, reclutaremos nuevos cosultorios de monitoreo.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: si crees que tus síntomas o problemas justifican un examen más detallado.
Perturbed: si crees que tus síntomas o problemas justifican un examen médico más detallado.

Original: En caso de respuesta afirmativa a estas preguntas de filtro, se debe pedir al paciente que no acuda al consultorio y que siga el esquema de PHE en su lugar.
Perturbed: En caso de respuesta afirmativa a estas preguntas de filtro, se debe pedir al paciente adulto que no acuda al consultorio y que siga el esquema de PHE en su lugar.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_intensifier_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words that changes the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Los síntomas comunes incluyen fiebre, tos seca y fatiga.
Perturbed: Los síntomas comunes incluyen fiebre alta, tos seca y fatiga.

Original: La dificultad para respirar, el dolor de garganta, los dolores de cabeza, las molestias corporales o la producción de esputo se encuentran entre los otros síntomas menos comunes.
Perturbed: La dificultad para respirar severa, el dolor de garganta, los dolores de cabeza, las molestias corporales o la producción de esputo se encuentran entre los otros síntomas menos comunes.

Original: ¿he trabajado en el cuerpo alrededor del pecho?
Perturbed: ¿he trabajado intensamente en el cuerpo alrededor del pecho?

Original: Se habían enviado formularios de informe a los CDC para 74 439 casos (60,7 %).
Perturbed: Se habían enviado formularios de informe a los CDC para un gran total de 74 439 casos (60,7 %).
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding words in the sentence. Please make sure that the added word does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Los síntomas comunes incluyen fiebre, tos seca y fatiga.
Perturbed: Los síntomas comunes incluyen fiebre y dolores musculares, tos seca y fatiga.

Original: La dificultad para respirar, el dolor de garganta, los dolores de cabeza, las molestias corporales o la producción de esputo se encuentran entre los otros síntomas menos comunes.
Perturbed: La dificultad para respirar, el dolor de garganta, los dolores de cabeza, las molestias corporales, la producción de esputo y trastornos digestivos se encuentran entre los otros síntomas menos comunes.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_omission_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by removing information from the sentence. Remove only one or two words from the sentence. Please make sure that the removed information does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Los síntomas comunes incluyen fiebre, tos seca y fatiga.
Perturbed: Los síntomas comunes incluyen fatiga y fatiga.

Original: Se están realizando investigaciones sobre una vacuna o un tratamiento antiviral específico.
Perturbed: Se están realizando investigaciones sobre un tratamiento antiviral específico.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_es = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the affirmative sentence into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: No logró aliviar el dolor con medicamentos que la competencia prohíbe a los participantes.
Perturbed: No logró aliviar el placer con medicamentos que la competencia prohíbe a los participantes.

Original: El mes pasado, un comité presidencial recomendó la renuncia del antiguo CEP como parte de medidas para llevar al país a nuevas elecciones.
Perturbed: El mes pasado, un comité presidencial no recomendó la renuncia del antiguo CEP como parte de medidas para llevar al país a nuevas elecciones.

Original: ¿y te gotea la nariz?
Perturbed: ¿y te canta la nariz?
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_synonym_hi = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: यह सचिवालय या क्लीनिकल स्टाफ के सदस्य हो सकते हैं, प्रत्येक कार्यालय की प्रक्रिया के अनुसार।
Perturbed: यह सचिवालय या चिकित्सा स्टाफ के सदस्य हो सकते हैं, प्रत्येक कार्यालय की प्रक्रिया के अनुसार।

Original: इसके अलावा, हम नए निगरानी कार्यालयों की भर्ती करेंगे।
Perturbed: इसके अलावा, हम नए अनुश्रवण कार्यालयों की भर्ती करेंगे।
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_word_order_hi = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the word order. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: यह सचिवालय या क्लीनिकल स्टाफ के सदस्य हो सकते हैं, प्रत्येक कार्यालय की प्रक्रिया के अनुसार।
Perturbed: यह क्लीनिकल स्टाफ या सचिवालय के सदस्य हो सकते हैं, प्रत्येक कार्यालय की प्रक्रिया के अनुसार।

Original: हम एक पर्यवेक्षण केंद्र विकसित करेंगे, जिसका उद्देश्य राष्ट्रीय स्तर पर डेटा प्रस्तुत करना और कार्यालयों को उनके डेटा की गुणवत्ता और विषाणु संबंधी व सीरोलॉजिकल नमूनों के संग्रह के बारे में टिप्पणी प्रदान करना है।
Perturbed: एक पर्यवेक्षण केंद्र राष्ट्रीय स्तर पर डेटा प्रस्तुत करने के लिए विकसित किया जाएगा, साथ ही एक डैशबोर्ड जो कार्यालयों को उनके डेटा की गुणवत्ता और नमूनों के संग्रह के बारे में टिप्पणी प्रदान करेगा।
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_hi = """Taks: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by making spelling of one or two words wrong. The words should be important words in the sentence but not words like "le", "et", "la" or "des". Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: यह सचिवालय या क्लीनिकल स्टाफ के सदस्य हो सकते हैं, प्रत्येक कार्यालय की प्रक्रिया के अनुसार।
Perturbed: यह सचिवाल्य या क्लीनिकल स्टाफ के सदस्य हो सकते हैं, प्रत्येक कार्याल्य की प्रक्रिया के अनुसार।

Original: इसके अलावा, हम नए निगरानी कार्यालयों की भर्ती करेंगे।
Perturbed: इसके अलावा, हम नए निगराणी कार्यलयों की भर्ती करेंगे।
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_hi = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: यदि आपको लगता है कि आपके लक्षण या समस्याएं अधिक विस्तृत परीक्षण की मांग करती हैं।
Perturbed: यदि आपको लगता है कि आपके लक्षण या स्वास्थ्य समस्याएं अधिक विस्तृत परीक्षण की मांग करती हैं।

Original: यदि इन स्क्रीनिंग प्रश्नों का उत्तर हां में है, तो रोगी को सलाह दी जानी चाहिए कि वह क्लिनिक न आए और इसके बजाय पीएचई आरेख का पालन करे।
Perturbed: यदि इन स्क्रीनिंग प्रश्नों का उत्तर हां में है, तो वयस्क रोगी को सलाह दी जानी चाहिए कि वह क्लिनिक न आए और इसके बजाय पीएचई आरेख का पालन करे।
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_intensifier_hi = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words that changes the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: सामान्य लक्षणों में बुखार, सूखी खांसी और थकान शामिल हैं।
Perturbed: सामान्य लक्षणों में तेज़ बुखार, सूखी खांसी और थकान शामिल हैं।

Original: सांस लेने में तकलीफ, गले में खराश, सिरदर्द, बदन दर्द या कफ बनना अन्य कम सामान्य लक्षणों में शामिल हैं।
Perturbed: सांस लेने में गंभीर तकलीफ, गले में खराश, सिरदर्द, बदन दर्द या कफ बनना अन्य कम सामान्य लक्षणों में शामिल हैं।

Original: क्या आपने छाती के आसपास शरीर पर काम किया?
Perturbed: क्या आपने विशेष रूप से छाती के आसपास शरीर पर काम किया?

Original: सीडीसी को 74,439 मामलों (60.7%) के लिए रिपोर्ट फॉर्म प्रस्तुत किए गए थे।
Perturbed: सीडीसी को कुल 74,439 मामलों (60.7%) के लिए रिपोर्ट फॉर्म प्रस्तुत किए गए थे।
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_hi = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding words in the sentence. Please make sure that the added word does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: सामान्य लक्षणों में बुखार, सूखी खांसी और थकान शामिल हैं।
Perturbed: सामान्य लक्षणों में बुखार और मांसपेशियों में दर्द, सूखी खांसी और थकान शामिल हैं।

Original: सांस लेने में तकलीफ, गले में खराश, सिरदर्द, बदन दर्द या कफ बनना अन्य कम सामान्य लक्षणों में शामिल हैं।
Perturbed: सांस लेने में तकलीफ, गले में खराश, सिरदर्द, बदन दर्द, कफ बनना और पाचन समस्याएं अन्य कम सामान्य लक्षणों में शामिल हैं।
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_omission_hi = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by removing information from the sentence. Remove only one or two words from the sentence. Please make sure that the removed information does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: सामान्य लक्षणों में बुखार, सूखी खांसी और थकान शामिल हैं।
Perturbed: सामान्य लक्षणों में थकान शामिल हैं।

Original: एक विशेष वैक्सीन या एंटीवायरल उपचार पर शोध चल रहा है।
Perturbed: एक एंटीवायरल उपचार पर शोध चल रहा है।
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_hi = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the affirmative sentence into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: वह दर्द को दवाओं से कम करने में सफल नहीं हुआ, जिन्हें प्रतियोगिता प्रतिभागियों के लेने पर रोक लगाती है।
Perturbed: वह आनंद को दवाओं से कम करने में सफल नहीं हुआ, जिन्हें प्रतियोगिता प्रतिभागियों के लेने पर रोक लगाती है।

Original: पिछले महीने, एक राष्ट्रपति समिति ने देश को नए चुनावों की ओर धकेलने के लिए पूर्व सीईपी के इस्तीफे की सिफारिश की।
Perturbed: पिछले महीने, एक राष्ट्रपति समिति ने देश को नए चुनावों की ओर धकेलने के लिए पूर्व सीईपी के इस्तीफे की सिफारिश नहीं की।

Original: और क्या आपकी नाक बह रही है?
Perturbed: और क्या आपकी नाक नाच रही है?
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_synonym_tl = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Maaari itong maging isang miyembro ng sekretarya o ng klinikal na kawani, ayon sa patakaran ng bawat klinika.
Perturbed: Maaari itong maging isang miyembro ng sekretarya o ng medikal na kawani, ayon sa patakaran ng bawat klinika.

Original: Bukod dito, magre-recruit kami ng mga bagong klinika para sa pagsubaybay.
Perturbed: Bukod dito, kukuha kami ng mga bagong klinika para sa monitoring.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_word_order_tl = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the word order. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Maaari itong maging isang miyembro ng sekretarya o ng klinikal na kawani, ayon sa patakaran ng bawat klinika.
Perturbed: Maaari itong maging isang miyembro ng klinikal na kawani o ng sekretarya, ayon sa patakaran ng bawat klinika.

Original: Magde-develop kami ng isang obserbatoryo na naglalayong ipakita ang datos sa pambansang antas pati na rin isang dashboard upang magbigay ng komento sa mga klinika tungkol sa kalidad ng kanilang datos at ang pagkolekta ng mga virological at serological samples.
Perturbed: Isang obserbatoryo ang ide-develop upang ipakita ang datos sa pambansang antas, kasama ang isang dashboard para magbigay ng komento sa mga klinika tungkol sa kalidad ng datos at ang pagkolekta ng mga virological at serological samples.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_tl = """Taks: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by making spelling of one or two words wrong. The words should be important words in the sentence but not words like "le", "et", "la" or "des". Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Maaari itong maging isang miyembro ng sekretarya o ng klinikal na kawani, ayon sa patakaran ng bawat klinika.
Perturbed: Maaari itong maging isang miyembro ng sekratarya o ng klinikal na kawni, ayon sa patakaran ng bawat klinika.

Original: Bukod dito, magre-recruit kami ng mga bagong klinika para sa pagsubaybay.
Perturbed: Bukod dito, magre-rekrut kami ng mga bagong klinika para sa pagsubaybay.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_tl = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Kung sa tingin mo ay nangangailangan ng masusing pagsusuri ang iyong mga sintomas o problema.
Perturbed: Kung sa tingin mo ay nangangailangan ng masusing medikal na pagsusuri ang iyong mga sintomas o problema.

Original: Kung ang sagot sa mga tanong sa pagsala ay oo, ang pasyente ay dapat sabihan na huwag pumunta sa klinika at sundin na lamang ang flowchart ng PHE.
Perturbed: Kung ang sagot sa mga tanong sa pagsala ay oo, ang pasyenteng adulto ay dapat sabihan na huwag pumunta sa klinika at sundin na lamang ang flowchart ng PHE.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_intensifier_tl = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words that changes the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Kasama sa mga karaniwang sintomas ang lagnat, tuyong ubo, at pagkapagod.
Perturbed: Kasama sa mga karaniwang sintomas ang mataas na lagnat, tuyong ubo, at matinding pagkapagod.

Original: Ang hirap sa paghinga, pananakit ng lalamunan, sakit ng ulo, pananakit ng katawan, o pag-ubo ay kabilang sa iba pang hindi gaanong karaniwang sintomas.
Perturbed: Ang matinding hirap sa paghinga, pananakit ng lalamunan, sakit ng ulo, pananakit ng katawan, o pag-ubo ay kabilang sa iba pang hindi gaanong karaniwang sintomas.

Original: Gumawa ba ako sa katawan sa paligid ng dibdib?
Perturbed: Gumawa ba ako nang matindi sa katawan sa paligid ng dibdib?

Original: Mga form ng ulat ay naisumite sa CDC para sa 74,439 kaso (60.7%).
Perturbed: Maraming mga form ng ulat ay naisumite sa CDC para sa 74,439 kaso (60.7%).
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_tl = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding words in the sentence. Please make sure that the added word does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Kasama sa mga karaniwang sintomas ang lagnat, tuyong ubo, at pagkapagod.
Perturbed: Kasama sa mga karaniwang sintomas ang lagnat, pananakit ng kalamnan, tuyong ubo, at pagkapagod.

Original: Ang hirap sa paghinga, pananakit ng lalamunan, sakit ng ulo, pananakit ng katawan, o pag-ubo ay kabilang sa iba pang hindi gaanong karaniwang sintomas.
Perturbed: Ang hirap sa paghinga, pananakit ng lalamunan, sakit ng ulo, pananakit ng katawan, pag-ubo, at mga problema sa pagtunaw ay kabilang sa iba pang hindi gaanong karaniwang sintomas.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_omission_tl = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by removing information from the sentence. Remove only one or two words from the sentence. Please make sure that the removed information does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Kasama sa mga karaniwang sintomas ang lagnat, tuyong ubo, at pagkapagod.
Perturbed: Kasama sa mga karaniwang sintomas ang pagkapagod.

Original: Isinasagawa ang mga pananaliksik para sa isang bakuna o tiyak na antiviral na paggamot.
Perturbed: Isinasagawa ang mga pananaliksik para sa isang tiyak na antiviral na paggamot.
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_tl = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the affirmative sentence into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: Hindi niya nagawang maibsan ang sakit gamit ang mga gamot na ipinagbabawal ng kumpetisyon sa mga kalahok.
Perturbed: Hindi niya nagawang maibsan ang kasiyahan gamit ang mga gamot na ipinagbabawal ng kumpetisyon sa mga kalahok.

Original: Noong nakaraang buwan, inirekomenda ng isang komiteng pangulo ang pagbibitiw ng dating CEP bilang bahagi ng mga hakbang upang itulak ang bansa sa mga bagong halalan.
Perturbed: Noong nakaraang buwan, hindi inirekomenda ng isang komiteng pangulo ang pagbibitiw ng dating CEP bilang bahagi ng mga hakbang upang itulak ang bansa sa mga bagong halalan.

Original: At tumutulo ba ang iyong ilong?
Perturbed: At sumasayaw ba ang iyong ilong?
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_synonym_zh = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by replacing one or two words (noun, verb, adjective or adverb) to its synonym. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 它可能是秘书处或临床工作人员的一员，根据每个诊所的规定。
Perturbed: 它可能是秘书处或医疗工作人员的一员，根据每个诊所的规定。

Original: 此外，我们将招聘新的监测诊所。
Perturbed: 此外，我们将雇用新的跟踪诊所。
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_word_order_zh = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the word order. Please make sure it does not changes the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 根据每个诊所的规定，这可能是秘书处或临床工作人员的一员。
Perturbed: 这可能是临床工作人员或秘书处的一员，根据每个诊所的规定。

Original: 我们将开发一个旨在展示全国数据的观测站，并提供一个仪表板以向诊所提供有关其数据质量和病毒及血清样本收集的反馈。
Perturbed: 一个旨在展示全国数据的观测站将被开发，并附带一个仪表板，用于向诊所提供有关其数据质量和病毒及血清样本收集的反馈。
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_spelling_zh = """Taks: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by making spelling of one or two words wrong. The words should be important words in the sentence but not words like "le", "et", "la" or "des". Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 根据每个诊所的规定，这可能是秘书处或临床工作人员的一员。
Perturbed: 根据每个诊所的规定，这可能是秘书处或临床工员的一员。

Original: 我们将开发一个旨在展示全国数据的观测站，并提供一个仪表板以向诊所提供有关其数据质量和病毒及血清样本收集的反馈。
Perturbed: 我们将开发一个旨在展示全国数据的观测站，并提供一个仪表版以向诊所提供有关其数据质量和病毒及血清样本收集的反馈。
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_noimpact_zh = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words in the sentence. Do not add words that change the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence and does not changes the meaning in a significant way. The added words would add more context that was already obvious from the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 如果您认为您的症状或问题需要更深入的检查。
Perturbed: 如果您认为您的症状或健康问题需要更深入的检查。

Original: 如果这些筛查问题的答案为“是”，患者应被建议不要去诊所，而是遵循 PHE 的流程图。
Perturbed: 如果这些筛查问题的答案为“是”，成年患者应被建议不要去诊所，而是遵循 PHE 的流程图。
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_intensifier_zh = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding one or two words that changes the intensity of the existing word. Please make sure that the added word does not disturb the grammaticality of the sentence. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 常见症状包括发烧、干咳和疲劳。
Perturbed: 常见症状包括高烧、干咳和严重的疲劳。

Original: 呼吸急促、喉咙痛、头痛、肌肉酸痛或痰液产生属于其他不太常见的症状。
Perturbed: 严重的呼吸急促、喉咙痛、头痛、肌肉酸痛或痰液产生属于其他不太常见的症状。

Original: 我在胸部周围做了按摩吗？
Perturbed: 我在胸部周围用力做了按摩吗？

Original: 向 CDC 提交了 74,439 例（60.7%）的报告表。
Perturbed: 向 CDC 提交了大量的 74,439 例（60.7%）的报告表。
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_expansion_impact_zh = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by adding words in the sentence. Please make sure that the added word does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 常见症状包括发烧、干咳和疲劳。
Perturbed: 常见症状包括发烧、肌肉疼痛、干咳和疲劳。

Original: 呼吸急促、喉咙痛、头痛、肌肉酸痛或痰液产生属于其他不太常见的症状。
Perturbed: 呼吸急促、喉咙痛、头痛、肌肉酸痛、痰液产生和消化问题属于其他不太常见的症状。
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_omission_zh = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by removing information from the sentence. Remove only one or two words from the sentence. Please make sure that the removed information does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 常见症状包括发烧、干咳和疲劳。
Perturbed: 常见症状包括疲劳。

Original: 关于疫苗或特定抗病毒治疗的研究正在进行中。
Perturbed: 关于特定抗病毒治疗的研究正在进行中。
*** Example Ends ***

Original: {{original}}
Perturbed: """


perturb_alteration_zh = """Task: You will be given a {{target_lang}} sentence and your goal is to perturb the sentence by changing the affirmative sentence into negation (vice versa) or changing one word (noun, verb, adjective or adverb) to its antonym or completely different word. Please make sure that the perturbation does not disturb the grammaticality of the sentence but should change the meaning in a significant way. Just output the perturbed {{target_lang}} sentence without giving any additional explanation.

*** Example Starts ***
Original: 他没有成功用药物缓解疼痛，这些药物是比赛禁止参赛者服用的。
Perturbed: 他没有成功用药物缓解快乐，这些药物是比赛禁止参赛者服用的。

Original: 上个月，总统委员会建议前CEP辞职，以推动该国举行新的选举。
Perturbed: 上个月，总统委员会没有建议前CEP辞职，以推动该国举行新的选举。

Original: 你的鼻子在流鼻涕吗？
Perturbed: 你的鼻子在跳舞吗？
*** Example Ends ***

Original: {{original}}
Perturbed: """



prompts = {
    # French
    "synonym_fr": perturb_synonym_fr,
    "word_order_fr": perturb_word_order_fr,
    "spelling_fr": perturb_spelling_fr,
    "expansion_noimpact_fr": perturb_expansion_noimpact_fr,
    "intensifier_fr": perturb_intensifier_fr,
    "expansion_impact_fr": perturb_expansion_impact_fr,
    "omission_fr": perturb_omission_fr,
    "alteration_fr": perturb_alteration_fr,

    # Spanish
    "synonym_es": perturb_synonym_es,
    "word_order_es": perturb_word_order_es,
    "spelling_es": perturb_spelling_es,
    "expansion_noimpact_es": perturb_expansion_noimpact_es,
    "intensifier_es": perturb_intensifier_es,
    "expansion_impact_es": perturb_expansion_impact_es,
    "omission_es": perturb_omission_es,
    "alteration_es": perturb_alteration_es,

    # Hindi
    "synonym_hi": perturb_synonym_hi,
    "word_order_hi": perturb_word_order_hi,
    "spelling_hi": perturb_spelling_hi,
    "expansion_noimpact_hi": perturb_expansion_noimpact_hi,
    "intensifier_hi": perturb_intensifier_hi,
    "expansion_impact_hi": perturb_expansion_impact_hi,
    "omission_hi": perturb_omission_hi,
    "alteration_hi": perturb_alteration_hi,

    # Tagalog
    "synonym_tl": perturb_synonym_tl,
    "word_order_tl": perturb_word_order_tl,
    "spelling_tl": perturb_spelling_tl,
    "expansion_noimpact_tl": perturb_expansion_noimpact_tl,
    "intensifier_tl": perturb_intensifier_tl,
    "expansion_impact_tl": perturb_expansion_impact_tl,
    "omission_tl": perturb_omission_tl,
    "alteration_tl": perturb_alteration_tl,

    # Chinese
    "synonym_zh": perturb_synonym_zh,
    "word_order_zh": perturb_word_order_zh,
    "spelling_zh": perturb_spelling_zh,
    "expansion_noimpact_zh": perturb_expansion_noimpact_zh,
    "intensifier_zh": perturb_intensifier_zh,
    "expansion_impact_zh": perturb_expansion_impact_zh,
    "omission_zh": perturb_omission_zh,
    "alteration_zh": perturb_alteration_zh,
}