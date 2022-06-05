import { ReactNode } from 'react'
import Link from 'next/link'
import { Button } from '../button/Button'
import { SidebarHeader } from '../shell/SidebarHeader'
import Dropdown from '../dropdown/dropdown'

type IShellProps = {
  title: string
  children: ReactNode
}

const Shell = (props: IShellProps) => {
  return (
    <SidebarHeader
      title={props.title}
      rightContent={
        <>
          <div className="flex">
            <div className="flex-1 mr-3">
              <Dropdown />
            </div>

            <div className="flex-1">
              <Link href="https://github.com/JakeValenzuela/ucfparking">
                <a>
                  <Button>GitHub</Button>
                </a>
              </Link>
            </div>
          </div>
        </>
      }
    >
      {props.children}
    </SidebarHeader>
  )
}
export { Shell }
