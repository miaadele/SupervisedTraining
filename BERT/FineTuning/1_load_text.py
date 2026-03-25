# This code loads the Geneva Bible dataset into a pandas DataFrame, 
# #prints summary information about the corpus, and 
# displays a small sample of verses so we can observe the Early Modern English language that we will later use to fine-tune BERT.
import pandas as pd

#Load the Geneva Bible as a pandas DataFrame
#each row is one verse organized by columns [‘Verse ID’, ‘Book Name’, ‘Book Number’, ‘Chapter’, ‘Verse’, ‘Text’].
df = pd.read_excel("genevaBible.xlsx")

print(f"Total verses: {len(df):,}")    #get the total number of verses
print(f"Columns: {list(df.columns)}")  #column names
print(f"\nBooks: {df['Book Name'].nunique()} unique books") #total number of books

print(f"First book: {df['Book Name'].iloc[0]}") # I bet you can figure these two out on your own!
print(f"Last book: {df['Book Name'].iloc[-1]}")

#Sample verses: Randomly select 10 verses from the dataset
print("\n=== Sample Verses ===\n")    
samples = df.sample(10, random_state=45)
for _, row in samples.iterrows():
    print(f"[{row['Book Name']} {row['Chapter']}:{row['Verse']}]")
    print(f"  {row['Text']}\n")

# Total verses: 31,102
# Columns: ['Verse ID', 'Book Name', 'Book Number', 'Chapter', 'Verse', 'Text']

# Books: 66 unique books
# First book: Genesis
# Last book: Revelation

# === Sample Verses ===

# [2 Samuel 16:18]
#   Hushai then answered vnto Absalom, Nay, but whome the Lord, and this people, and all the men of Israel chuse, his will I be, and with him will I dwell.

# [Jeremiah 52:27]
#   And the king of Babel smote them, and slewe them in Riblah, in the lande of Hamath: thus Iudah was caried away captiue out of his owne land.

# [Deuteronomy 25:9]
#   Then shall his kinswoman come vnto him in the presence of the Elders, and loose his shooe from his foote, and spit in his face, and answere, and say, So shall it be done vnto that man, that will not buylde vp his brothers house.

# [Jeremiah 31:36]
#   If these ordinances depart out of my sight, saith the Lorde, then shall the seede of Israel cease from being a nation before me, for euer.

# [John 5:21]
#   For likewise as the Father rayseth vp the dead, and quickeneth them, so the Sonne quickeneth whom he will.

# [Ezra 2:54]
#   The sonnes of Neziah, the sonnes of Hatipha,

# [Romans 9:3]
#   For I woulde wish my selfe to be separate from Christ, for my brethren that are my kinsemen according to the flesh,

# [1 John 2:21]
#   I haue not written vnto you, because ye knowe not the trueth: but because ye knowe it, and that no lie is of the trueth.

# [Deuteronomy 28:48]
#   Therefore thou shalt serue thine enemies which the Lord shal send vpon thee, in hunger & in thirst, and in nakednesse, and in neede of all things? And he shall put a yoke of yron vpon thy necke vntill he haue destroyed thee.

# [1 Samuel 23:7]
#   And it was tolde Saul that Dauid was come to Keilah, & Saul sayd, God hath deliuered him into mine hand: for he is shut in, seeing he is come into a citie that hath gates and barres.