import pandas as pd
import networkx as nx

def corr_for_col(data: pd.DataFrame, col_name: str) -> pd.DataFrame:
    """
    Compute correlation matrix for a specific column across all tickers.

    Args:
        data (pd.DataFrame): Multi-index DataFrame with stock data.
        col_name (str): Column name to compute correlation for (e.g., "OC_next", "CO_next").

    Returns:
        pd.DataFrame: Correlation matrix (absolute values).
    """
    # Select all tickers under the specified top-level column
    df_col = data.loc[:, (col_name, slice(None))].copy()
    
    # Flatten columns for convenience
    df_col.columns = [t for _, t in df_col.columns]

    # Compute absolute correlation
    corr = df_col.corr().abs()
    
    return corr

def get_most_correlated_pairs(corr: pd.DataFrame, top_x: int = 1):
    """
    For each stock, pick its top X most correlated stocks and form unique sets.
    
    Parameters
    ----------
    corr : pd.DataFrame
        Absolute correlation matrix (tickers x tickers)
    top_x : int
        Number of closest correlated stocks to include per stock.
    
    Returns
    -------
    List[set]
        List of unique sets of correlated tickers.
    """
    tickers = corr.columns
    added_sets = []

    for t in tickers:
        # Get correlations of this stock with all others, sorted descending
        sorted_corr = corr[t].sort_values(ascending=False)
        # Exclude self
        sorted_corr = sorted_corr[sorted_corr.index != t]

        count = 0
        for other_ticker in sorted_corr.index:
            pair_set = {t, other_ticker}

            # Check if this set already exists
            if not any(pair_set == s for s in added_sets):
                added_sets.append(pair_set)
                count += 1

            if count >= top_x:
                break

    return added_sets

def graph_from_pairs(top_pairs, corr = None):
    """
    Create a graph from pairs of correlated stocks.

    Parameters
    ----------
    pairs : List[set]
        List of sets of correlated tickers.

    Returns
    -------
    networkx.Graph
        Graph where nodes are tickers and edges represent correlations.
    """
    G = nx.Graph()

    # Add edges for each pair
    for pair in top_pairs:
        stock1, stock2 = tuple(pair)
        # Use correlation as edge weight (optional)
        if corr is not None:
            weight = corr.loc[stock1, stock2]
        G.add_edge(stock1, stock2, weight=weight)

    return G

def connect_components_by_highest_corr(G, corr):
    """
    Connect disconnected components of G by adding edges with the highest correlation
    until the graph is fully connected.
    """
    while not nx.is_connected(G):
        components = list(nx.connected_components(G))
        # Find the pair of nodes across different components with highest correlation
        max_corr = -1
        best_pair = None
        
        for i in range(len(components)):
            for j in range(i+1, len(components)):
                comp_i = components[i]
                comp_j = components[j]
                for node_i in comp_i:
                    for node_j in comp_j:
                        if corr.loc[node_i, node_j] > max_corr:
                            max_corr = corr.loc[node_i, node_j]
                            best_pair = (node_i, node_j)
        
        # Add the edge with highest correlation
        if best_pair:
            G.add_edge(*best_pair, weight=max_corr)

    return G

def generate_edges_list(data, col_name, top_x=1):
    corr = corr_for_col(data, col_name)
    top_pairs = get_most_correlated_pairs(corr, top_x=top_x)
    G = graph_from_pairs(top_pairs, corr=corr)
    G_connected = connect_components_by_highest_corr(G, corr)
    edges_list = list(G_connected.edges())

    return edges_list


def top_bottom_edges_list(data,outputs = ["OC_next", "CO_next"],top_x=1):
    """
    Generate edges list based on both top and bottom correlations of "Close" and "Volume".

    Parameters
    ----------
    data : pd.DataFrame
        Multi-index DataFrame with stock data.
    top_x : int
        Number of closest correlated stocks to include per stock.

    Returns
    -------
    List[tuple]
        List of edges representing correlations.
    """
    edges_list = []
    for output in outputs:
        edges_list.append(generate_edges_list(data, output, top_x=top_x))

    return edges_list
