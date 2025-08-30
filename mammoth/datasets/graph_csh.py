from mammoth.datasets.dataset import Dataset


class Graph_CSH(Dataset):
    def __init__(self, papers_df, affiliations_df, sensitive_columns=[]):
        self.papers_df = papers_df
        self.affiliations_df = affiliations_df
        self.G = None
        self.cols = sensitive_columns

    def create_coauth_graph(self):
        import networkx as nx
        import numpy as np

        # Creating co-authorship network:
        Authors_papers_links = {}
        Papers_authors_links = {}

        DF_Affiliations = self.affiliations_df

        for i in self.affiliations_df.index:
            try:
                Authors_papers_links[DF_Affiliations["researcher_id"][i]].add(
                    DF_Affiliations["doi"][i]
                )
            except:
                Authors_papers_links[DF_Affiliations["researcher_id"][i]] = set(
                    [DF_Affiliations["doi"][i]]
                )

            try:
                Papers_authors_links[DF_Affiliations["doi"][i]].add(
                    DF_Affiliations["researcher_id"][i]
                )
            except:
                Papers_authors_links[DF_Affiliations["doi"][i]] = set(
                    [DF_Affiliations["researcher_id"][i]]
                )

        Co_authorship_links = {}

        for k in Authors_papers_links.keys():
            Co_authorship_links[k] = {}

        for doi in Papers_authors_links.keys():
            authors = Papers_authors_links[doi]
            for k1 in authors:
                for k2 in authors:
                    if k1 != k2:
                        try:
                            Co_authorship_links[k1][k2]["papers"].add(doi)
                        except:
                            Co_authorship_links[k1][k2] = {"papers": set([doi])}

                        try:
                            Co_authorship_links[k2][k1]["papers"].add(doi)
                        except:
                            Co_authorship_links[k2][k1] = {"papers": set([doi])}

        # Compute the weights of the network:
        for u in Co_authorship_links.keys():
            for v in Co_authorship_links[u].keys():
                Co_authorship_links[u][v]["weight"] = len(
                    Co_authorship_links[u][v]["papers"]
                )

        Couthorship_network = nx.Graph(Co_authorship_links)
        Couthorship_network.remove_node(np.nan)

        # Create values of citations, productivity, and degree:
        for u in Couthorship_network.nodes():
            Filtered_DF = self.papers_df[
                self.papers_df.doi.isin(Authors_papers_links[u])
            ]

            # 0. Add protected attributes:
            max_year = self.papers_df.loc[Filtered_DF.year.idxmax()]["year"]
            Filtered_affiliation = DF_Affiliations[
                (DF_Affiliations.year == max_year)
                & (DF_Affiliations.researcher_id == u)
            ]
            for attribute in ["Nationality", "aff_country", "Gender"]:
                Couthorship_network.nodes[u][attribute] = Filtered_affiliation.iloc[0][
                    attribute
                ]

                if attribute != "Gender":
                    for category in ["Region", "IncomeGroup"]:
                        Couthorship_network.nodes[u][attribute + "_" + category] = (
                            Filtered_affiliation.iloc[0][attribute + "_" + category]
                        )

            # 1. Analyse the rankings:
            Citations = sum(Filtered_DF.N_citations)
            Productivity = len(Authors_papers_links[u])
            Degree = sum(
                [
                    Couthorship_network[u][v]["weight"]
                    for v in Couthorship_network[u].keys()
                ]
            )
            Couthorship_network.nodes[u]["Citations"] = Citations
            Couthorship_network.nodes[u]["Productivity"] = Productivity
            Couthorship_network.nodes[u]["Degree"] = Degree

        self.G = Couthorship_network

    def return_num_nodes(self):
        return self.papers_df.shape[0]

    def return_num_edges(self):
        return self.affiliations_df.shape[0]
