# Checkout Runbook

## Error E-447

Error code E-447 means checkout failed during payment authorization.

First check payment gateway status.
Then inspect the payments-service logs.
If the gateway is healthy, verify that checkout-api is sending a valid customer token.

## Checkout Service Flow

The web app calls checkout-api.
checkout-api calls payments-service.
payments-service writes payment results to orders-db.
The support assistant can retrieve similar prior tickets from the vector database.
