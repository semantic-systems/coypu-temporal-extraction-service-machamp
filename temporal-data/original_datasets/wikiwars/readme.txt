2010-06-01
Pawel Mazur, Pawel.Mazur@science.mq.edu.au
Robert Dale, Robert.Dale@science.mq.edu.au


0. Licence

All the documents in the corpus are sources from English Wikipedia. In consequence, the corpus is
released under the Creative Commons Attribution-ShareAlike 3.0 License - see the licence at
http://en.wikipedia.org/wiki/Wikipedia:Text_of_Creative_Commons_Attribution-ShareAlike_3.0_Unported_License.

See also the Terms of Use at http://wikimediafoundation.org/wiki/Terms_of_Use.

Important part is that each time 'you distribute or publicly perform the work or a collection, the 
licensor offers to the recipient a license to the work on the same terms and conditions as the 
license granted to you under this license.'


1. Collecting Data

First, a query to Google "most famous wars in history" returned as the top result a Yahoo Answers 
page titled ‘10 most famous wars in history?’ (located at 
http://answers.yahoo.com/question/index?qid=20090209222618AAauMN1, last accessed on 2009-12-20).
It listed the wars presented in the table below, in which we also provide the years of the wars and 
links to Wikipedia articles.

Name of the war				Year span		URL
World War II				1939 – 1945		en.wikipedia.org/wiki/World_War_II
World War I					1914 – 1918		en.wikipedia.org/wiki/World_War_I
American Civil War			1861 – 1865		en.wikipedia.org/wiki/American_Civil_War
American Revolution			1775 – 1783		en.wikipedia.org/wiki/American_Revolutionary_War
Vietnam War					1955 – 1975		en.wikipedia.org/wiki/Vietnam_War
Korean War					1950 – 1953		en.wikipedia.org/wiki/Korean_War
Iraq War					2003 – ...		en.wikipedia.org/wiki/Iraq_War
French Revolution			1789 – 1799		en.wikipedia.org/wiki/French_Revolution
Persian Wars				499 – 450 BC	en.wikipedia.org/wiki/Greco-Persian_Wars
Punic Wars					264 – 146 BC	en.wikipedia.org/wiki/Punic_Wars

We then queried Google for "the biggest wars" and again we chose the top result, 
which was a page titled ‘The 20 biggest wars of the 20th century’ (located at 
http://users.erols.com/mwhite28/war-list.htm, last accessed on 2009-12-20).
It listed the biggest wars in the 20th century measured by the number of military victims. Names of 
these wars are presented in the table below along with years and links to Wikipedia articles about 
them.

Name of the war				Year span		URL
World War II				1939 – 1945		en.wikipedia.org/wiki/World_War_II
World War I					1914 – 1918		en.wikipedia.org/wiki/World_War_I
Korean War					1950 – 1953		en.wikipedia.org/wiki/Korean_War
Chinese Civil War			1945 – 1949		en.wikipedia.org/wiki/Chinese_Civil_War
Vietnam War					1955 – 1975		en.wikipedia.org/wiki/Vietnam_War
Iran-Iraq War				1980 – 1988		en.wikipedia.org/wiki/IranIraq_War
Russian Civil War			1917 – 1923		en.wikipedia.org/wiki/Russian_Civil_War
Chinese Civil War			1927 – 1937		en.wikipedia.org/wiki/Chinese_Civil_War
French Indochina War		1946 – 1954		en.wikipedia.org/wiki/First_Indochina_War
Mexican Revolution			1911 – 1920		en.wikipedia.org/wiki/Mexican_Revolution
Spanish Civil War			1936 – 1939		en.wikipedia.org/wiki/Spanish_Civil_War
French-Algerian War			1954 – 1962		en.wikipedia.org/wiki/Algerian_War
Soviet-Afghanistan War		1979 – 1989		en.wikipedia.org/wiki/Soviet_war_in_Afghanistan
Russo-Japanese War			1904 – 1905		en.wikipedia.org/wiki/Russo-Japanese_War
Riffian War					1921 – 1926		no article found
First Sudanese Civil War	1955 – 1972		en.wikipedia.org/wiki/First_Sudanese_Civil_War
Russo-Polish War			1919 – 1921		en.wikipedia.org/wiki/PolishSoviet_War
Biafran War					1967 – 1970		en.wikipedia.org/wiki/Nigerian_Civil_War
Chaco War					1932 – 1935		en.wikipedia.org/wiki/Chaco_War
Abyssinian War				1935 – 1936		en.wikipedia.org/wiki/Second_Italo-Abyssinian_War

We then combined the two lists, eliminating duplicates which resulted in 25 links (note that the 
second list considered two periods of the Chinese Civil War separately; because both these events 
are described at the same URL in Wikipedia and because there were also other periods of the civil 
war in China in the 20th century, we treated all these periods as part of one war). As we were 
unable to find any article on Wikipedia on the Riffian War, we dropped it from the list. 
We also considered two articles (about the First Sudanese Civil War and Chaco War) to be too short, 
with a little number of temporal expressions and not presenting the course of the events, and in 
consequence to be not interesting and useful for our experiments. For this reason we dropped them 
too. This resulted in 22 articles included in the corpus.

2. Text Extraction and Preprocessing

To prepare the corpus, we first manually copied text from those sections of the webpages that 
described the course of the wars. This involved manual removal of picture captions and cross-page 
links. We then ran a script over the results of this extraction process to convert some Unicode 
characters into ASCII (ligatures, spaces, apostrophes, hyphens and other punctuation marks), 
and to remove citation links and a variety of other Wikipedia annotations. Finally, we converted 
each of the text files into an SGML file: each document was wrapped in one DOC tag, inside which 
there are DOCID, DOCTYPE and DATETIME tags. The document time stamp is the date and time at which we
downloaded the page from Wikipedia to our local repository. The proper content of the article is 
wrapped in a TEXT tag. This document structure intentionally follows that of the ACE 2005 and 2007 
documents, so as to make the processing and evaluation of the WikiWars data highly compatible with 
the tools used to process the ACE corpora.

3. The Annotation Process

Having prepared the input SGML documents, we then processed them with the DANTE temporal expression 
tagger (see [1]). DANTE outputs the original SGML documents augmented with an inline TIMEX2 
annotation for each temporal expression found. These output files can be imported to Callisto [2],
an annotation tool that supports TIMEX2 annotations. Using a temporal expression tagger as a 
first-pass annotation tool not only significantly reduces the amount of human annotation effort 
required (creating a tag from scratch requires a number of clicks in the annotation tool), but also 
helps to minimize the number of errors that arise from overlooking markable expressions through 
'annotator blindness'. The annotations produced by DANTE were then manually corrected in Callisto 
via the following process. First, Annotator #1 corrected all the annotations produced by DANTE, both
in terms of extent and the values provided for TIMEX2 attributes. This process also included the 
annotation of any temporal expression missed by the automatic tagger, and the removal of spurious 
matches. Then, Annotator #2 checked all the revised annotations and prepared a list of errors found 
and doubts or queries in regard to potentially problematic annotations. Annotator #1 then verified 
and fixed the errors, after discussion in the case of disagreements.
The final SGML files containing inline annotations were then transformed into ACE APF XML annotation
files, this being the stand-off markup format developed for ACE evaluations. This transformation was
carried out using the tern2apf [3] tool developed by NIST for the ACE 2004 evaluations, with some 
modifications introduced by us to adjust the tool to support ACE 2005 documents and to add a 
document ID as part of the ID of a TIMEX2 annotation (so that all annotations would have corpus-wide
unique IDs). The resulting gold standard annotations are thus available in two formats: one contains 
the original documents enriched with inline annotations, and the other consists of stand-off 
annotations in the ACE APF format.

4. Corpus Statistics

The corpus contains 22 documents with a total of almost 120,000 tokens (these were counted using 
GATE’s default English tokeniser; hyphenated words, e.g. 'British-held' and 'co-operation', were 
treated as single tokens. For more information on GATE see [4].) and 2,671 temporal expressions 
annotated in the TIMEX2 (September 2005) format [5]. Below are statistics on the individual 
documents that make up the corpus:

Document ID					Tokens		TIMEX2		Tokens/TIMEX2
01_WW2						5,593		  169		33.1
02_WW1						10,370		  264		39.3
03_AmCivWar					3,529		   75		47.1
04_AmRevWar					5,695		  146		39.0
05_VietnamWar				11,640		  243		47.9
06_KoreanWar				5,992		  147		40.8
07_IraqWar					8,404		  247		34.0
08_FrenchRev				9,631		  174		55.4
09_GrecoPersian				7,393		  129		57.3
10_PunicWars				3,475		   57		61.0
11_ChineseCivWar			3,905		  103		37.9
12_IranIraq					4,508		   98		46.0
13_RussianCivWar			3,924		  103		38.1
14_FirstIndochinaWar		3,085		   70		44.1
15_MexicanRev				3,910		   77		50.8
16_SpanishCivilWar			1,455		   63		23.1
17_AlgerianWar				7,716		  130		59.4
18_SovietsInAfghanistan		5,306		  110		48.2
19_RussoJap					2,760		   62		44.5
20_PolishSoviet				5,137		  106		48.5
21_NigerianCivilWar			2,091		   29		72.1
22_2ndItaloAbyssinianWar	3,949		   69		57.2
----------------------------------------------------
Total for the whole corpus	119,468		2,671		44.7
Average per document		5,430		  121	 	 –
Standard deviation			2,663		   63		 –


5. Acknowledgements

In the course of preparation of this corpus we used the GATE framework (see http://gate.ac.uk)
and the Callisto tool (see http://callisto.mitre.org).

If you publish any work concerning WikiWars and would like to cite this resource, please use
the following details:

Pawel Mazur and Robert Dale [2010] WikiWars: A New Corpus for Research on Temporal Expressions.
In the Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP),
9-11 October, Massachusetts, USA,


6. References

[1] Pawel Mazur and Robert Dale [2007] The DANTE Temporal Expression Tagger. 
In Zygmunt Vetulani, editor, Proceedings of the 3rd Language And Technology Conference (LTC), 
October, Poznan, Poland.

[2] http://callisto.mitre.org

[3] http://www.itl.nist.gov/iad/mig//tests/ace/2004/software.html

[4] http://gate.ac.uk

[5] http://timex2.mitre.org/annotation_guidelines/timex2_annotation_guidelines.html


