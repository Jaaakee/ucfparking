"""Schedule a cron job running the scraper."""
from apscheduler.schedulers.blocking import BlockingScheduler
from main import main

sched = BlockingScheduler()


@sched.scheduled_job("cron", hour="*")
def timed_job():
    """Run the main function every hour."""
    main()


sched.start()
