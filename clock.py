from apscheduler.schedulers.blocking import BlockingScheduler

from main import main

sched = BlockingScheduler()


@sched.scheduled_job('cron', hour='*', minute=0, second=0)
def timed_job():
    main()


sched.start()
