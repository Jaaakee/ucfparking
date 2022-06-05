import { ReactNode } from 'react'

type ISidebarHeaderProps = {
  title: string
  rightContent: ReactNode
  children: ReactNode
}

const SidebarHeader = (props: ISidebarHeaderProps) => {
  return (
    <div className="antialiased text-gray-600 flex h-screen">
      <div className="bg-primary-100 flex-1 flex flex-col overflow-hidden">
        <header className="dark:bg-[#292b2e] h-16 bg-white flex items-center justify-between py-3 px-3 sm:px-5 lg:px-6">
          <div className="text-lg font-bold dark:text-gray-400 text-gray-900 ">
            {props.title}
          </div>

          <div className="ml-auto">{props.rightContent}</div>
        </header>

        <div className="dark:bg-[#202124] flex-1 overflow-y-auto overflow-x-hidden">
          <div>{props.children}</div>
        </div>
      </div>
    </div>
  )
}

export { SidebarHeader }
