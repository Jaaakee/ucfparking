"""Schedule a cron job running the scraper."""
from apscheduler.schedulers.blocking import BlockingScheduler
from cache_endpoints import cache_results
from main import main

sched = BlockingScheduler()


@sched.scheduled_job("cron", hour="*")
def timed_job():
    """Run the main function every hour."""
    main()
    cache_results("https://api.ucfparking.com")


sched.start()
