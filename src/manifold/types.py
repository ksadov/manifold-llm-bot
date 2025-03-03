from typing import Optional
from enum import Enum
from typing import Optional


class OutcomeType(Enum):
    BINARY = "BINARY"
    FREE_RESPONSE = "FREE_RESPONSE"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    NUMERIC = "NUMERIC"
    PSEUDO_NUMERIC = "PSEUDO_NUMERIC"
    BOUNTIED_QUESTION = "BOUNTIED_QUESTION"
    POLL = "POLL"


class AddAnswersMode(Enum):
    ANYONE = "ANYONE"
    ONLYCREATOR = "ONLY_CREATOR"
    DISABLED = "DISABLED"


class Option:
    text: str
    votes: int


class LiteMarket:
    def __init__(
        self,
        id: str,
        creatorId: str,
        creatorUsername: str,
        creatorName: str,
        createdTime: int,
        question: str,
        url: str,
        outcomeType: OutcomeType,
        mechanism: str,
        totalLiquidity: float,
        volume: float,
        volume24Hours: float,
        isResolved: bool,
        uniqueBettorCount: int,
        creatorAvatarUrl: Optional[str] = None,
        closeTime: Optional[int] = None,
        p: Optional[dict] = None,
        probability: Optional[float] = None,
        pool: Optional[dict] = None,
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        isLogScale: Optional[bool] = None,
        resolutionTime: Optional[int] = None,
        resolution: Optional[str] = None,
        resolutionProbability: Optional[float] = None,
        lastUpdatedTime: Optional[int] = None,
        lastBetTime: Optional[int] = None,
        marketTier: Optional[str] = None,
        siblingContractId: Optional[str] = None,
        slug: Optional[str] = None,
        token: Optional[str] = None,
        lastCommentTime: Optional[int] = None,
        resolverId: Optional[str] = None,
        comments: Optional[list[str]] = None,
    ):
        self.id = id
        self.creatorId = creatorId
        self.creatorUsername = creatorUsername
        self.creatorName = creatorName
        self.creatorAvatarUrl = creatorAvatarUrl
        self.createdTime = createdTime
        self.closeTime = closeTime
        self.question = question
        self.url = url
        self.outcomeType = OutcomeType(outcomeType)
        self.mechanism = mechanism
        self.probability = probability
        self.pool = pool
        self.p = p
        self.totalLiquidity = totalLiquidity
        self.value = value
        self.min = min
        self.max = max
        self.isLogScale = isLogScale
        self.volume = volume
        self.volume24Hours = volume24Hours
        self.isResolved = isResolved
        self.resolutionTime = resolutionTime
        self.resolution = resolution
        self.resolutionProbability = resolutionProbability
        self.uniqueBettorCount = uniqueBettorCount
        self.lastUpdatedTime = lastUpdatedTime
        self.lastBetTime = lastBetTime
        self.marketTier = marketTier
        self.siblingContractId = siblingContractId
        self.slug = slug
        self.token = token
        self.lastCommentTime = lastCommentTime
        self.resolverId = resolverId
        self.comments = comments

    def __repr__(self):
        return f"LiteMarket(id={self.id}, question={self.question})"

    def __str__(self):
        return self.__repr__()


class FullMarket(LiteMarket):
    def __init__(
        self,
        id: str,
        creatorId: str,
        creatorUsername: str,
        creatorName: str,
        createdTime: int,
        question: str,
        url: str,
        outcomeType: OutcomeType,
        mechanism: str,
        totalLiquidity: float,
        volume: float,
        volume24Hours: float,
        isResolved: bool,
        uniqueBettorCount: int,
        description: dict,
        textDescription: str,
        creatorAvatarUrl: Optional[str] = None,
        closeTime: Optional[int] = None,
        p: Optional[dict] = None,
        probability: Optional[float] = None,
        pool: Optional[dict] = None,
        value: Optional[float] = None,
        min: Optional[float] = None,
        max: Optional[float] = None,
        isLogScale: Optional[bool] = None,
        resolutionTime: Optional[int] = None,
        resolution: Optional[str] = None,
        resolutionProbability: Optional[float] = None,
        lastUpdatedTime: Optional[int] = None,
        lastBetTime: Optional[int] = None,
        marketTier: Optional[str] = None,
        siblingContractId: Optional[str] = None,
        slug: Optional[str] = None,
        token: Optional[str] = None,
        lastCommentTime: Optional[int] = None,
        resolverId: Optional[str] = None,
        answers: Optional[list[str]] = None,
        shouldAnswersSumToOne: Optional[bool] = None,
        addAnswersMode: Optional[AddAnswersMode] = None,
        options: Optional[list[Option]] = None,
        totalBounty: Optional[float] = None,
        bountyLeft: Optional[float] = None,
        coverImageUrl: Optional[str] = None,
        groupSlugs: Optional[list[str]] = None,
        comments: Optional[list[str]] = None,
    ):
        super().__init__(
            id,
            creatorId,
            creatorUsername,
            creatorName,
            createdTime,
            question,
            url,
            outcomeType,
            mechanism,
            probability,
            pool,
            totalLiquidity,
            volume,
            volume24Hours,
            isResolved,
            uniqueBettorCount,
            creatorAvatarUrl,
            closeTime,
            p,
            value,
            min,
            max,
            isLogScale,
            resolutionTime,
            resolution,
            resolutionProbability,
            lastUpdatedTime,
            lastBetTime,
            marketTier,
            siblingContractId,
            slug,
            token,
            lastCommentTime,
            resolverId,
            comments,
        )
        self.description = description
        self.textDescription = textDescription
        self.answers = answers
        self.shouldAnswersSumToOne = shouldAnswersSumToOne
        self.addAnswersMode = addAnswersMode
        self.options = options
        self.totalBounty = totalBounty
        self.bountyLeft = bountyLeft
        self.coverImageUrl = coverImageUrl
        self.groupSlugs = groupSlugs

    def __repr__(self):
        return f"FullMarket(id={self.id}, question={self.question}, description={self.textDescription})"

    def __str__(self):
        return self.__repr__()


class Fees:
    def __init__(
        self,
        platformFee: float,
        liquidityFee: float,
        creatorFee: float,
    ):
        self.platformFee = platformFee
        self.liquidityFee = liquidityFee
        self.creatorFee = creatorFee

    def __repr__(self):
        return f"Fees(platformFee={self.platformFee}, liquidityFee={self.liquidityFee}, creatorFee={self.creatorFee})"

    def __str__(self):
        return self.__repr__()


class Fill:
    def __init__(
        self,
        amount: int,
        matchedBetId: str,
        shares: float,
        timestamp: int,
    ):
        self.amount = amount
        self.matchedBetId = matchedBetId
        self.shares = shares
        self.timestamp = timestamp

    def __repr__(self):
        return f"Fill(amount={self.amount}, matchedBetId={self.matchedBetId}, timestamp={self.timestamp})"

    def __str__(self):
        return self.__repr__()


class Bet:
    def __init__(
        self,
        amount: int,
        contractId: str,
        orderAmount: int,
        shares: float,
        isFilled: bool,
        isCancelled: bool,
        fills: list[Fill],
        outcome: str,
        probBefore: float,
        probAfter: float,
        loanAmount: int,
        createdTime: int,
        isRedemption: bool,
        visibility: str,
        betId: str,
        fees: Fees,
        limitProb: Optional[float] = None,
        expiresMillisAfter: Optional[int] = None,
        expiresAt: Optional[int] = None,
    ):
        self.amount = amount
        self.contractId = contractId
        self.orderAmount = orderAmount
        self.shares = shares
        self.isFilled = isFilled
        self.isCancelled = isCancelled
        self.fills = fills
        self.outcome = outcome
        self.probBefore = probBefore
        self.probAfter = probAfter
        self.loanAmount = loanAmount
        self.createdTime = createdTime
        self.isRedemption = isRedemption
        self.visibility = visibility
        self.expiresAt = expiresAt
        self.betId = betId
        self.fees = fees
        self.limitProb = limitProb
        self.expiresMillisAfter = expiresMillisAfter

    def __repr__(self):
        return f"Bet(amount={self.amount}, contractId={self.contractId}, outcome={self.outcome}, createdTime={self.createdTime})"

    def __str__(self):
        return self.__repr__()


class User:
    def __init__(
        self,
        id: str,
        createdTime: int,
        name: str,
        username: str,
        url: str,
        balance: int,
        totalDeposits: float,
        lastBetTime: Optional[int] = None,
        currentBettingStreak: Optional[int] = None,
        avatarUrl: Optional[str] = None,
        bio: Optional[str] = None,
        bannerUrl: Optional[str] = None,
        website: Optional[str] = None,
        twitterHandle: Optional[str] = None,
        discordHandle: Optional[str] = None,
        isBot: Optional[bool] = None,
        isAdmin: Optional[bool] = None,
        isTrustworthy: Optional[bool] = None,
        isBannedFromPosting: Optional[bool] = None,
        userDeleted: Optional[bool] = None,
        verifiedPhone: Optional[bool] = None,
        creatorTraders: Optional[int] = None,
        signupBonusPaid: Optional[bool] = None,
        isBannedFromMana: Optional[bool] = None,
        nextLoanCached: Optional[int] = None,
        hasSeenLoanModal: Optional[bool] = None,
        isAdvancedTrader: Optional[bool] = None,
        kycDocumentStatus: Optional[str] = None,
        optOutBetWarnings: Optional[bool] = None,
        shouldShowWelcome: Optional[bool] = None,
        streakForgiveness: Optional[int] = None,
        followerCountCached: Optional[int] = None,
        isBannedFromSweepcash: Optional[bool] = None,
        cashBalance: Optional[int] = None,
        spiceBalance: Optional[int] = None,
        totalCashDeposits: Optional[float] = None,
    ):
        self.id = id
        self.createdTime = createdTime
        self.name = name
        self.username = username
        self.url = url
        self.balance = balance
        self.totalDeposits = totalDeposits
        self.lastBetTime = lastBetTime
        self.currentBettingStreak = currentBettingStreak
        self.avatarUrl = avatarUrl
        self.bio = bio
        self.bannerUrl = bannerUrl
        self.website = website
        self.twitterHandle = twitterHandle
        self.discordHandle = discordHandle
        self.isBot = isBot
        self.isAdmin = isAdmin
        self.isTrustworthy = isTrustworthy
        self.isBannedFromPosting = isBannedFromPosting
        self.userDeleted = userDeleted
        self.verifiedPhone = verifiedPhone
        self.creatorTraders = creatorTraders
        self.signupBonusPaid = signupBonusPaid
        self.isBannedFromMana = isBannedFromMana
        self.nextLoanCached = nextLoanCached
        self.hasSeenLoanModal = hasSeenLoanModal
        self.isAdvancedTrader = isAdvancedTrader
        self.kycDocumentStatus = kycDocumentStatus
        self.optOutBetWarnings = optOutBetWarnings
        self.shouldShowWelcome = shouldShowWelcome
        self.streakForgiveness = streakForgiveness
        self.followerCountCached = followerCountCached
        self.isBannedFromSweepcash = isBannedFromSweepcash
        self.cashBalance = cashBalance
        self.spiceBalance = spiceBalance
        self.totalCashDeposits = totalCashDeposits

    def __repr__(self):
        return f"User(id={self.id}, name={self.name}, username={self.username})"

    def __str__(self):
        return self.__repr__()
