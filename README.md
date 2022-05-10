
# UCF Parking

This project makes it easy to view parking garage space availability on the University of Central Florida's main campus. This project contains a web scraper and REST API written in Python and a site written in Next.js with Tailwind CSS (ucfparking.com). This project aims to relieve the stress of finding a parking spot on campus and assist students with creating a plan before arriving on campus.
## Features
- 🚀 Hourly parking data scraped since March 21st, 2021!
- 🖥️ Public API with access to a bunch of methods. [Learn more!](#api-routes)
- 📦 Out of the box included dashboard with various customizable sorting and viewing options.
- 📱 Mobile support!
## Installation
Before running, there are a few environment variables in this project that I use to deploy easily:
- `DB_HOST` - your database's host address
- `DB_USER` - your database's user name
- `DB_PASS` - your database's user pass
- `DB_NAME` - your database's name
- `DB_PORT` - your database's port . . . *usually 3306!*
- `SENTRY_URL` - [very helpful debugging tool](https://sentry.io/)

With this current project setup, a MySQL database is required. I use a plan provided by [ClearDB MySQL on Heroku](https://devcenter.heroku.com/articles/cleardb).

Once you've set that all up, you can begin to start scraping data and setup your own API!

To run the web scraping script, run the following command inside the project:
```bash
  python api/deploy/main.py
```

To get the API to work, take a look at the [Procfile](https://github.com/JakeValenzuela/ucfparking/blob/master/Procfile) and how we use [Uvicorn](https://www.uvicorn.org/) to run it.
## Deployment
The API, data scraper, and database currently run on a dyno on Heroku. The Next.js site is deployed on Vercel.

This project uses [Advanced Python Scheduler](https://apscheduler.readthedocs.io/) to run the data scrapper every sharp hour (:00).

The Heroku add-on's that are used include: [ClearDB MySQL](https://devcenter.heroku.com/articles/cleardb) and [New Relic's APM](https://newrelic.com/products/application-monitoring).
## API Routes
Please note that these routes are for [ucfparking.com](https://ucf-parking-data.herokuapp.com). Please note that each of these routes returns a page of JSON. The timezone in this project is UTC. Queries with an asterik (*) next to the description may take a bit of time depending on the amount of data in the database.

#### Get the last row of data entered in the database.
[`/`](https://ucf-parking-data.herokuapp.com/)

#### * Get all of the data in the database.
[`/all`](https://ucf-parking-data.herokuapp.com/all)

#### Get the combined data from the past 24 hours.
[`/today`](https://ucf-parking-data.herokuapp.com/today)

#### Get the combined data from the past 7 days.
[`/week`](https://ucf-parking-data.herokuapp.com/week)

#### * Get access to the specified year's data.
[`/:year:`](https://ucf-parking-data.herokuapp.com/2022)

#### Get access to the specified year-month's data.
[`/:year:/:month:`](https://ucf-parking-data.herokuapp.com/2022/01)

#### Get access to the specified year-month-day's data.
[`/:year:/:month:/:day:`](https://ucf-parking-data.herokuapp.com/2022/01/01)

#### Get statistics about the data.
[`/stats`](https://ucf-parking-data.herokuapp.com/stats)
