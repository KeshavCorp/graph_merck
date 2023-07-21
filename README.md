# graph_merck

This code essentilly takes input a CSV file of data which should have columns like short and long description, type, source url etc. 
Each of these rows will be treated as a separate node for graph creation. 

The end goal here will be to create a graph which showcases the interconnected nodes via the text in the rows.
For encoding, a sentence transformer will be used to encode the text and cosine similarity is used to find the distance between them.

The similarity matrix can be filtered based on upper and lower level threshold in the code.

Happy graphing! 
