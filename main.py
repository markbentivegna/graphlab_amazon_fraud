from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T

SOURCE=f"/mnt/raid0_ssd_8tb/dataset/Amazon"
instruments_meta = "meta_Musical_Instruments.json"
instruments_reviews= "reviews_Musical_Instruments.json"
meta_filename = f"{SOURCE}/Category/{instruments_meta}"
review_filename = f"{SOURCE}/Category/{instruments_reviews}"


spark = SparkSession.builder.config("spark.executor.memory", "128g") \
.config("spark.driver.memory", "128g") \
.config("spark.memory.offHeap.size","128g") \
.config("spark.memory.offHeap.enabled",True) \
.appName("graphlab_amazon_fraud").getOrCreate()

meta_df = spark.read.json(meta_filename)
review_df = spark.read.json(review_filename)

equals_zero = lambda e: e == 0
# review_df = review_df.withColumn("drop", F.exists(F.col("helpful"), equals_zero)) \
#     .filter(F.col("drop") == False).drop(F.col("drop")) \
#     .select("asin", "overall", "reviewText", "reviewerID", "reviewerName", 
#         "summary", review_df.helpful[0], review_df.helpful[1], "unixReviewTime", "reviewTime"
#     ).groupBy("reviewerID").agg(
#         F.sum("helpful[0]").alias("helpful_0"),
#         F.sum("helpful[1]").alias("helpful_1"),
#         F.collect_list("asin").alias("asin"),
#         F.collect_list("overall").alias("overall"),
#         F.collect_list("reviewText").alias("reviewText"),
#         F.collect_list("summary").alias("summary"),
#         F.collect_list("reviewerName").alias("reviewerName"),
#         F.collect_list("unixReviewTime").alias("unixReviewTime"),
#         F.collect_list("reviewTime").alias("reviewTime")
#     ).withColumns({
#         "reviewLen": F.udf(lambda a: len(a), T.IntegerType())(F.col("reviewerName")),
#         "reviewSetLen": F.udf(lambda a: len(set(a)), T.IntegerType())(F.col("reviewerName")),
#     }).drop(
#         F.col("reviewLen")).drop(F.col("reviewSetLen")
#     )

# U-P-U : it connects users reviewing at least one same product
upu_groupings = review_df.groupBy("asin").agg(F.collect_list("reviewerID").alias("reviewerID")).collect()



def format_date(review_time):
    month, day, year = review_time.replace(",","").split(" ")
    return f"{int(month):02d}-{int(day):02d}-{int(year):04d}"

# U-S-U : it connects users having at least one same star rating within one week
format_date_udf = F.udf(lambda x: format_date(x), T.StringType())
review_df = review_df.withColumn("formatted_date", F.to_date(format_date_udf(F.concat_ws(",", "reviewTime")), "MM-dd-yyyy"))
usu_groupings = review_df.groupBy(F.weekofyear("formatted_date").alias("week"), "overall").agg(F.collect_list("reviewerID").alias("reviewerID")).collect()


# U-V-U : it connects users with top 5% mutual review text similarities (measured byTF-IDF) among all users.
# TODO: generate text similarites by review content

# refined_reviews_file = "Musical_Instruments_Reviews"
# review_df.coalesce(1).write.format('json').save(refined_reviews_file)