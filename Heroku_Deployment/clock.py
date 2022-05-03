from apscheduler.schedulers.blocking import BlockingScheduler
from api.main import main

sched = BlockingScheduler()


@sched.scheduled_job('cron', hour='*')
def timed_job():
    main()


sched.start()
