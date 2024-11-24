# Machine Learning?!

This repository contains some of my efforts to learn how machine learning works on a more technical level.

## `embedding-tester.py`

This script programmatically clusters requirements based on their semantic similarity. This might be useful for drafting application architecture given a list of plain-text requirements.

The idea is to demonstrate two machine-learning techniques:
- Sentence-level embedding
- Semantic clustering

## Setup

- Start with Python 3.
- Install the necessary Python libraries as follows:

```sh
python -m venv .
. bin/activate
pip install sentence-transformers faiss-cpu scikit-learn numpy
```

- Alternatively, run the setup script (does exactly what you see in the shell script above)
- `setup-venv-and-dependencies.sh`

## Usage

The list of requirements lives in the file `early-bird-requirements.txt`. These requirements come from the "EarlyBird Case Study" as provided to students at the FH Technikum Wien for the Software Architecture module.
The script reads this file, generates sentence-level embeddings, puts them into a vector database, then clusters the sentences. The `n_clusters` variable can be changed as needed.

The script downloads model files the first time they're used, meaning the first run of the script will take much longer than subsequent runs.

Example usage:
``` sh
$ cd /path/to/this/repository/
$ . /bin/activate
(embeddings-and-clustering) $ python embedding-tester.py
```

Example output:
```
Generating embeddings for the requirements...
Embedding Shape: (41, 768)
Number of embeddings in FAISS index: 41

Clustered Requirements:

Cluster 1: Order, number, customers
  - Customers place orders exclusively via phone.
  - Customers provide a seven-digit customer number (including area code and checksum).
  - Phone clerks verify the customer's status (e.g., not blacklisted for bad payment behavior).
  - Customers can directly name products when ordering.
  - Customers receive an eight-digit order number with a checksum after placing an order.
  - Customers can inquire about order status using their order number.
  - Customers can order via text messages by sending a predefined string to the company number.
  - The system will reply to text orders with the assigned order number.
  - Payment records will include customer number, order number, amount, and payment due date.
  - The business report will include orders, products, amounts, clerks, customers, addresses, and order numbers.

Cluster 2: Delivery, clerks, packing
  - Breakfast must be delivered in under 25 minutes to all city areas.
  - Offer predefined breakfasts (e.g., mini-breakfast, luxury breakfast).
  - Each customer has one predefined address, eliminating the need to specify a delivery address.
  - Packing clerks assemble products and place them in paper bags.
  - Packing clerks create labels with their name, customer’s name, address, order number, and delivery clerk.
  - Two invoices are printed for each order, showing label data, products, amounts, prices, and total cost.
  - Delivery clerks optimize routes using a spreadsheet and print them.
  - Delivery clerks deliver bags and invoices, with customers signing one copy and keeping the other.
  - Customers, packing clerks, delivery clerks, and managers will use the web application.
  - Customers confirm delivery by entering a password in a browser on the delivery clerk’s smartphone.
  - The payment system will receive records for expected payments after packing completion.

Cluster 3: Web, application, specified
  - A web application will automate existing processes without altering them.
  - The web application will replace phone ordering, label text processing, and route planning tools.
  - The web application must support specified browsers.
  - The system will include a browser-based, unauthenticated product search feature.
  - A nightly business report will be generated automatically for managers.

Cluster 4: Products, blueprint, order
  - Customers can customize their breakfast from a list of simple products (e.g., croissants, orange juice).
  - Prepackaged products can contain simple products or other prepackaged products.
  - An order consists of a variety of simple and/or prepackaged products.
  - Each product has a unit (e.g., grams) and a price per unit (in Euros).
  - Customers can choose products from a list based on criteria (e.g., calories, price).
  - Customers can use a previous order as a blueprint for new orders.
  - Each order can reference one blueprint at most.
  - A blueprint can be reused multiple times.
  - Each reprinted invoice gets a unique number.
  - Each product has a unique product code.

Cluster 5: Assembly, orders, canceled
  - Orders can be canceled before assembly.
  - Orders cannot be canceled after assembly.
  - Cancellations cannot be undone.
  - Order updates are not allowed; customers must cancel and place a new order.
  - Order cancellations can be made by sending a specific text string to the company number.
```
