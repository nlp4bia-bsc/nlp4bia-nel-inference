"""
Named Entity Linking (NEL) Inference Module

This module provides functionality for performing Named Entity Linking using
dense retrieval methods with sentence transformers.
"""

import sys
import os
import pandas as pd
import torch
from nlp4bia.linking.retrievers import DenseRetriever
from sentence_transformers import SentenceTransformer
from optparse import OptionParser


def run_nel_inference(gazetteer,
                      input_file,
                      output_file,
                      model,
                      k=10,
                      store_vector_db=None,
                      vector_db_file='vector_db.pt',
                      input_mentions=None,
                      save_output=True):
    """
    Run Named Entity Linking inference to link mentions to gazetteer entries.
    
    This function takes input mentions and links them to entries in a gazetteer
    using dense vector similarity. It can either load pre-computed vectors or
    compute them on the fly.
    
    Parameters
    ----------
    gazetteer : str
        Path to the gazetteer file (TSV format) containing terms and their codes
    input_file : str
        Path to the input file (TSV format) containing mentions to be linked
    output_file : str
        Path where the output file will be saved
    model : str
        Name or path of the sentence transformer model to use for encoding
    k : int, optional
        Number of top candidates to retrieve for each mention (default: 10)
    store_vector_db : str, optional
        Path to save the computed vector database for future reuse (default: None)
    vector_db_file : str, optional
        Path to load pre-computed vector database (default: 'vector_db.pt')
    input_mentions : list, optional
        List of mentions to process instead of reading from input_file (default: None)
    save_output : bool, optional
        Whether to save the output to a file (default: True)
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing the input mentions with their top-k candidate links,
        including codes, terms, and similarity scores
    """
    
    # Load the sentence transformer model
    print('Loading model...')
    st_model = SentenceTransformer(model)
    print('Model loaded.')

    # Load gazetteer data
    gazetteer_df = pd.read_csv(gazetteer, sep='\t')
    
    # Process input mentions
    if input_mentions is None:
        # Read mentions from input file if no direct list provided
        input_df = pd.read_csv(input_file, sep='\t')
        mentions = input_df.span.unique().tolist()
    else:
        # Use provided mentions list
        mentions = input_mentions
        input_df = pd.DataFrame({"span": mentions})

    # Extract terms from gazetteer for encoding
    terms = gazetteer_df["term"].tolist()

    # Set device for computation (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load or compute vector database for gazetteer terms
    if os.path.exists(vector_db_file):
        print("Loading vector database from file...")
        vector_db = torch.load(vector_db_file, map_location=device)
    else:
        print("Vector database not found. Computing vector database...")
        vector_db = st_model.encode(
            terms,
            show_progress_bar=True, 
            convert_to_tensor=True,
            batch_size=4096,
            device=device.type  # Ensure encoding runs on the same device
        )

    # Save vector database if requested
    if store_vector_db is not None:
        torch.save(vector_db, store_vector_db)
        print(f"Vector database stored at {store_vector_db}")

    # Initialize dense retriever with gazetteer and vector database
    biencoder = DenseRetriever(df_candidates=gazetteer_df, vector_db=vector_db, model_or_path=st_model)

    # Retrieve top-k candidates for each mention
    candidates = biencoder.retrieve_top_k(
        mentions, 
        k=k, 
        input_format="text",
        return_documents=True
    )
    
    # Convert results to DataFrame and merge with input
    candidates_df = pd.DataFrame(candidates).rename(columns={"mention": "span"})
    candidates_df = input_df.merge(candidates_df, on="span", how="left")
    
    # Format similarity scores to 4 decimal places
    candidates_df["similarity"] = candidates_df["similarity"].apply(
        lambda sims: [round(s, 4) for s in sims] if isinstance(sims, list) else round(sims, 4)
    )
    candidates_df = candidates_df.rename(columns={"similarity": "similarities"})
    
    # Reorganize columns for better readability
    cols_to_move = ["span", "codes", "terms", "similarities"]
    new_order = [col for col in candidates_df.columns if col not in cols_to_move] + cols_to_move
    output = candidates_df[new_order]
    
    # Save output if requested
    if save_output:
        output.to_csv(output_file, sep="\t", index=False)
        print(f"Output saved to {output_file}")
    
    return output


def main(argv=None):
    """
    Main function to handle command-line interface for NEL inference.
    
    Parses command-line arguments and runs the NEL inference pipeline.
    
    Parameters
    ----------
    argv : list, optional
        Command-line arguments. If None, uses sys.argv[1:] (default: None)
    """
    # Set up command-line argument parser
    parser = OptionParser(description="Named Entity Linking Inference Tool")
    
    # Define command-line options
    parser.add_option("-g", "--gazetteer", dest="gazetteer", 
                     help="Gazetteer file path, tab-separated values extension (.tsv)", 
                     default="gazetteer.tsv")
    parser.add_option("-i", "--input", dest="input_file", 
                     help="Input file path, tab-separated values extension (.tsv)", 
                     default="input.tsv")
    parser.add_option("-o", "--output", dest="output_file", 
                     help="Output file path, tab-separated values extension (.tsv)",
                     default="output.tsv")
    parser.add_option("-m", "--model", dest="model", 
                     help="Sentence transformer model to be used for encoding", 
                     default="ICB-UMA/ClinLinker-KB-GP")
    parser.add_option("-k", "--top_k", dest="k", type="int", 
                     help="Number of top candidates to retrieve (default: 10)", 
                     default=10)
    parser.add_option("-s", "--store_vector_db", dest="store_vector_db", 
                     help="Path to store the vector database for future reuse", 
                     default=None)
    parser.add_option("-v", "--vector_db_file", dest="vector_db_file", 
                     help="Path to load pre-computed vector database (default: 'vector_db.pt')", 
                     default='vector_db.pt')
    
    # Parse command-line arguments
    (options, args) = parser.parse_args(argv)

    # Run NEL inference with parsed options
    run_nel_inference(
        gazetteer=options.gazetteer,
        input_file=options.input_file,
        output_file=options.output_file,
        model=options.model,
        k=options.k,
        store_vector_db=options.store_vector_db,
        vector_db_file=options.vector_db_file
    )


if __name__ == "__main__":
    # Entry point: run main function and exit with appropriate status code
    sys.exit(main())