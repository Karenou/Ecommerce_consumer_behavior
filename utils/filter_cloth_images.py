import pandas as pd
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--product_csv", type=str, help="csv path of product info")
    parser.add_argument("--image_csv", type=str, help="csv path of image index")
    parser.add_argument("--filter_csv", type=str, help="save the filter cloth image csv")
    args = parser.parse_args()

    product = pd.read_csv(args.product_csv)
    product = product[["sku_id", "primary_category_name_en"]]

    # filter cloth skus
    product = product[product["primary_category_name_en"].str.contains("Accessories") == False]
    product = product[product["primary_category_name_en"].str.contains("Jewelry") == False]
    product = product[product["primary_category_name_en"].str.contains("Bags") == False]
    product = product[product["primary_category_name_en"].str.contains("Socks") == False]
    product = product[product["primary_category_name_en"].str.contains("Toys") == False]
    product = product[product["primary_category_name_en"].str.contains("Shoes") == False]
    product.to_csv("category.csv", index=False)

    # keep at most two images per sku
    image = pd.read_csv(args.image_csv)

    result = product.merge(image, on="sku_id", how="left")
    result.to_csv(args.filter_csv, index=False)




