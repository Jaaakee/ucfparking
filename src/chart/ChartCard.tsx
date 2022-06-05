import { ReactElement } from 'react'

import { ResponsiveContainer } from 'recharts'

type IChartCardProps = {
  title: string
  children: ReactElement
}

const ChartCard = (props: IChartCardProps) => (
  <div className="dark:bg-[#292b2e] bg-white border dark:border-gray-800 border-gray-200 rounded-md pt-6 pb-8">
    <div className="text-lg font-semibold dark:text-gray-400 text-gray-800 pl-5 mb-8">
      {props.title}
    </div>

    <ResponsiveContainer height={400}>{props.children}</ResponsiveContainer>
  </div>
)

export { ChartCard }
