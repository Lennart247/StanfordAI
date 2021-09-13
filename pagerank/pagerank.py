import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    #chance = random.uniform(0, 1)
    newDict = {}
    randomProb = 1 - damping_factor
    if len(corpus[page]) == 0:
        for randomPage in corpus:
            newDict[randomPage] = 1 / len(corpus)
    else:
        for randomPage in corpus:
            newDict[randomPage] = randomProb*1/len(corpus)
        for linkedPage in corpus[page]:
            newDict[linkedPage] = newDict[linkedPage]+(1-randomProb)/len(corpus[page])
    return newDict


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRankDict = corpus.copy()
    currentPage = random.sample(pageRankDict.keys(), 1)[0]
    for page in corpus:
        pageRankDict[page] = 0.0
    # Inverse CDF Funktion? Nein. random.choices kann Verteilungen benutzen.
    pageRankDict[currentPage] = 1/n
    distribution = transition_model(corpus, currentPage, damping_factor)
    for i in range(1, n):
        nextPage = random.choices(list(distribution.keys()), list(distribution.values()))[0]
        pageRankDict[nextPage] = pageRankDict[nextPage] + 1/n
        distribution = transition_model(corpus, nextPage, damping_factor)
    return pageRankDict


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    threshold = 0.001
    pageRankDict = corpus.copy()
    for key in pageRankDict.keys():
        pageRankDict[key] = 1/len(corpus)

    while True:
        oldValues = pageRankDict.copy() # PYTHON NUTZT REFERENZEN!
        not_changed = 0

        for site in pageRankDict.keys():
            linkSum = 0.0
            for page in corpus.keys():
                if site in corpus[page]:
                    linkSum = linkSum + pageRankDict[page]/len(corpus[page])

            pageRankDict[site] = (1-damping_factor)/len(corpus) + damping_factor * linkSum
            if abs(pageRankDict[site] - oldValues[site]) < threshold:
                not_changed += 1
        if not_changed == len(corpus):
            break
    return pageRankDict



if __name__ == "__main__":
    main()
