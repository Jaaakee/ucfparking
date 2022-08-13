"""Schedule a cron job running the scraper."""
from apscheduler.schedulers.blocking import BlockingScheduler
from main import main
from utils.fill_missing_dates import fill_missing_dates_main

sched = BlockingScheduler()


@sched.scheduled_job("cron", hour="*")
def timed_job():
    """Run the main function every hour."""
    main()
    fill_missing_dates_main()


sched.start()
