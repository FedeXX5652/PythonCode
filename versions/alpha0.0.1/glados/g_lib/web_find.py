def _web_find_(query):
    from googlesearch import search
    print("This is what I foud for "+query+":")
    for j in search(query, tld="com", lang='en', num=1, stop=3, pause=2): 
        print(j)