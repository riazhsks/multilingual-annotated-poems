import pandas as pd

languages = ["English", "C3P", "CCV", "German", "Spanish", "Hungarian"]

sets = {"English" : {"s1":set(),"s2":set(),"s3":set(),"s4":set(),"s5":set(),"s6":set(),"s7":set(),"s8":set()}, 
        "C3P" : {"s1":set(),"s2":set(),"s3":set(),"s4":set(),"s5":set(),"s6":set(),"s7":set(),"s8":set()}, 
        "CCV" : {"s1":set(),"s2":set(),"s3":set(),"s4":set(),"s5":set(),"s6":set(),"s7":set(),"s8":set()},
        "German" : {"s1":set(),"s2":set(),"s3":set(),"s4":set(),"s5":set(),"s6":set(),"s7":set(),"s8":set()}, 
        "Spanish" : {"s1":set(),"s2":set(),"s3":set(),"s4":set(),"s5":set(),"s6":set(),"s7":set(),"s8":set()}, 
        "Hungarian" : {"s1":set(),"s2":set(),"s3":set(),"s4":set(),"s5":set(),"s6":set(),"s7":set(),"s8":set()}}
results = []
for lang in languages:
    for set_num in range(1,9):
        line_id = 1
        with open (f"predictions/samples/{lang}/set{set_num}_classification.txt", "r") as f:
            for line in f:
                form = line.split(" ")[0].strip()
                sets[lang][f"s{set_num}"].add(form)
                record = {"Poem ID": line_id, "Corpus": lang, f"S{set_num}" : form}
                results.append(record)
                line_id +=1
      
df = pd.DataFrame(results)
df = df.groupby(['Corpus', 'Poem ID'], as_index=False).first()

df.to_csv(f"rule_based_preds.csv", index=False)
